#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <chrono>
using namespace std;

// ------------------------------- Config -------------------------------
struct Config {
    string index_dir = "index_output";
    string invlists_bin = "invlists.bin";
    string lastdocid_bin = "lastdocid.bin";
    string docidsize_bin = "docidsize.bin";
    string freqsize_bin  = "freqsize.bin";
    string lexicon_tsv   = "lexicon.tsv";
    string collection_tsv = "collection.tsv"; // optional: docID \t ... \t length
    size_t topk = 10;
    bool conjunctive = false;  // default OR
    float k1 = 1.2f; // BM25 parameter Term Frequency Saturation
    float b  = 0.75f; // BM25 parameter Length Normalization Factor
} CFG;

// ------------------------- Small utils / IO ---------------------------
static inline bool fileExists(const string& p) {
    ifstream f(p, ios::binary); return f.good();
}
static inline string pathJoin(const string& a, const string& b) {
    if (a.empty()) return b;
    if (a.back()=='/' || a.back()=='\\') return a + b;
    return a + "/" + b;
}

// ------------------------------- VarByte ------------------------------
// VarByte format from BuildIndex: buf[n-1] first (< 128), then buf[n-2..0] with MSB=1 (>= 128)
// So: [high_7bits < 128], [mid_7bits | 0x80], ..., [low_7bits | 0x80]
// Example: 128 = [1, 128]: (1 << 7) | (128 & 0x7F) = 128 + 0 = 128
// Example: 300 = [2, 172]: (2 << 7) | (172 & 0x7F) = 256 + 44 = 300
struct VarByte {
    // decode one uint32 from ptr (advance ptr)
    static inline uint32_t decode(const uint8_t *&p) {
        // First byte always has MSB=0 (value 0-127)
        uint8_t b = *p++;
        uint32_t x = b;
        
        // Continue reading while we see bytes with MSB=1 (>= 128)
        // These are the continuation bytes
        while (*p >= 128) {
            b = *p++;
            x = (x << 7) | (b & 0x7F);
        }
        return x;
    }
    
    // encode one uint32 to a vector (match BuildIndex format)
    static inline void encode(uint32_t v, vector<uint8_t>& out) {
        uint8_t buf[10];
        int n = 0;
        do {
            buf[n++] = (uint8_t)(v & 0x7F);
            v >>= 7;
        } while (v != 0);
        // Write last byte first (MSB=0), then others with MSB=1
        out.push_back(buf[n - 1]);
        for (int i = n - 2; i >= 0; --i) {
            out.push_back(buf[i] | 0x80);
        }
    }
};

// ------------------------- Loaded Metadata ----------------------------
struct Meta {
    vector<uint32_t> lastDoc;     // size = #blocks
    vector<uint32_t> docBytes;    // size = #blocks
    vector<uint32_t> freqBytes;   // size = #blocks
    vector<uint64_t> blockByteOff; // prefix sum of (docBytes+freqBytes), for block i
    uint64_t totalBytes = 0;

    void buildOffsets() {
        blockByteOff.resize(docBytes.size());
        uint64_t acc = 0;
        for (size_t i=0;i<docBytes.size();++i) {
            blockByteOff[i] = acc;
            acc += (uint64_t)docBytes[i] + (uint64_t)freqBytes[i];
        }
        totalBytes = acc;
    }
};

static vector<uint32_t> readU32File(const string& filePath) {
    ifstream in(filePath, ios::binary);
    if (!in) throw runtime_error("Cannot open: " + filePath);
    in.seekg(0, ios::end);
    size_t bytes = (size_t)in.tellg();
    in.seekg(0);
    if (bytes % 4 != 0) throw runtime_error("Corrupt .bin (not multiple of 4): " + filePath);
    size_t n = bytes / 4;
    vector<uint32_t> v(n);
    in.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

// ------------------------------- Lexicon ------------------------------
struct LexiconEntry {
    string term;
    uint32_t startBlock = 0; // inclusive
    uint32_t endBlock   = 0; // inclusive
    uint32_t postings   = 0; // ft
    uint64_t offset     = 0; // byte offset into invlists.bin where the term list begins (optional)
};

// tries to auto-detect numeric columns in line tokens
static bool parseLexiconLine(const vector<string>& tok, LexiconEntry& e) {
    // expect at least 3 columns; typically term and several numeric fields
    if (tok.size() < 3) return false;
    e.term = tok[0];

    // collect numeric columns
    vector<long double> nums;
    vector<int> idx;
    for (int i=1;i<(int)tok.size();++i) {
        try {
            size_t pos=0;
            long double v = stold(tok[i], &pos);
            if (pos == tok[i].size()) { nums.push_back(v); idx.push_back(i); }
        } catch (...) {}
    }
    if (nums.empty()) return false;

    // Heuristic:
    // - postings (ft) is reasonable (>=1), blocks are integers and start<=end, offset is large-ish.
    // We'll map by count:
    //   4 numeric: [startBlock, endBlock, postings, offset] (order may vary)
    //   3 numeric: [startBlock, endBlock, postings] (offset omitted)
    // Fallback: guess by relative magnitude.
    auto asU32 = [&](long double v){ return (uint32_t) llround(v); };
    auto asU64 = [&](long double v){ return (uint64_t) llround(v); };

    if (nums.size() >= 4) {
        // sort by value to guess: smallest ~startBlock, next ~endBlock, then postings, largest ~offset
        vector<pair<long double,int>> p;
        for (int k=0;k<(int)nums.size();++k) p.push_back({nums[k], k});
        // But postings may be < blocks count; better: choose largest as offset (very likely bytes)
        int k_offset = max_element(nums.begin(), nums.end()) - nums.begin();
        // remove that, then among remaining, choose min as startBlock, max as endBlock, leftover as postings
        vector<int> rem;
        for (int k=0;k<(int)nums.size();++k) if (k != k_offset) rem.push_back(k);
        int k_min = rem[0], k_max = rem[0];
        for (int k: rem) { if (nums[k] < nums[k_min]) k_min = k; if (nums[k] > nums[k_max]) k_max = k; }
        // leftover:
        int k_post = -1; for (int k: rem) if (k != k_min && k != k_max) { k_post = k; break; }
        e.startBlock = asU32(nums[k_min]);
        e.endBlock   = asU32(nums[k_max]);
        e.postings   = k_post>=0 ? asU32(nums[k_post]) : 0;
        e.offset     = asU64(nums[k_offset]);
        return true;
    } else if (nums.size() == 3) {
        // assume startBlock, endBlock, postings
        // smallest=start, largest=end, middle=postings
        int k_min=0, k_max=0;
        for (int k=1;k<3;++k) { if (nums[k] < nums[k_min]) k_min=k; if (nums[k] > nums[k_max]) k_max=k; }
        int k_mid = 3 - k_min - k_max;
        e.startBlock = asU32(nums[k_min]);
        e.endBlock   = asU32(nums[k_max]);
        e.postings   = asU32(nums[k_mid]);
        e.offset     = 0;
        return true;
    }
    return false;
}

static unordered_map<string, LexiconEntry> loadLexicon(const string& filePath) {
    ifstream in(filePath);
    if (!in) throw runtime_error("Cannot open lexicon: " + filePath);
    unordered_map<string, LexiconEntry> L;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        // split by tab/space
        vector<string> tok; tok.reserve(8);
        {
            string tmp; tmp.reserve(line.size());
            for (char c : line) {
                if (c=='\t' || c==' ') { if (!tmp.empty()){ tok.push_back(tmp); tmp.clear(); } }
                else tmp.push_back(c);
            }
            if (!tmp.empty()) tok.push_back(tmp);
        }
        LexiconEntry e;
        if (parseLexiconLine(tok, e)) { L.emplace(e.term, e); }
    }
    if (L.empty()) throw runtime_error("Lexicon parsed 0 entries. Check format.");
    return L;
}

// ------------------------ Document lengths (optional) -----------------
struct DocStats {
    vector<uint32_t> len;   // doc length (tokens) by docID (0-based)
    uint64_t totalLen = 0;
    uint32_t N = 0;
    double avgdl = 1.0;
};

static DocStats loadDocStatsOptional(const string& collection_tsv) {
    DocStats ds; ds.N=0; ds.avgdl = 1.0;
    if (!fileExists(collection_tsv)) return ds; // will fill later on first query if needed
    ifstream in(collection_tsv);
    if (!in) return ds;
    string line;
    // Expect at least: docID<TAB>...<TAB>length
    // We will detect docID as first integer on the line and last integer as length.
    uint32_t maxDoc = 0;
    vector<pair<uint32_t,uint32_t>> pairs;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> tok; tok.reserve(16);
        {
            string t;
            for (char c : line) {
                if (c=='\t') { if(!t.empty()){ tok.push_back(t); t.clear(); } }
                else t.push_back(c);
            }
            if (!t.empty()) tok.push_back(t);
        }
        // find first and last integers
        int firstI=-1, lastI=-1;
        for (int i=0;i<(int)tok.size();++i) {
            try { stoll(tok[i]); if (firstI==-1) firstI=i; lastI=i; } catch(...) {}
        }
        if (firstI!=-1 && lastI!=-1 && firstI!=lastI) {
            uint32_t id = (uint32_t)stoul(tok[firstI]);
            uint32_t ln = (uint32_t)stoul(tok[lastI]);
            pairs.emplace_back(id, ln);
            maxDoc = max(maxDoc, id);
        }
    }
    if (pairs.empty()) return ds;
    ds.len.assign(maxDoc+1, 0);
    for (auto &pr : pairs) { ds.len[pr.first] = pr.second; ds.totalLen += pr.second; }
    ds.N = (uint32_t)ds.len.size();
    ds.avgdl = ds.N ? (double)ds.totalLen / (double)ds.N : 1.0;
    if (ds.avgdl <= 0.0) ds.avgdl = 1.0;
    return ds;
}

// ------------------------------- IndexReader --------------------------
class IndexReader {
public:
    explicit IndexReader(const Config& cfg)
    : cfg_(cfg) {
        // load metadata
        meta_.lastDoc = readU32File(pathJoin(cfg_.index_dir, cfg_.lastdocid_bin));
        meta_.docBytes= readU32File(pathJoin(cfg_.index_dir, cfg_.docidsize_bin));
        meta_.freqBytes=readU32File(pathJoin(cfg_.index_dir, cfg_.freqsize_bin));
        if (meta_.docBytes.size() != meta_.freqBytes.size()
         || meta_.docBytes.size() != meta_.lastDoc.size()) {
            throw runtime_error("Metadata arrays sizes differ");
        }
        meta_.buildOffsets();

        // map invlists file
        invPath_ = pathJoin(cfg_.index_dir, cfg_.invlists_bin);
        inv_.open(invPath_, ios::binary);
        if (!inv_) throw runtime_error("Cannot open invlists: " + invPath_);

        // load lexicon
        L_ = loadLexicon(pathJoin(cfg_.index_dir, cfg_.lexicon_tsv));
    }

    struct BlockData {
        vector<uint32_t> docIDs; // absolute docIDs for this block
        vector<uint32_t> freqs;
    };

    struct ListIter {
        const LexiconEntry* lex = nullptr;
        uint32_t curBlock = 0;
        uint32_t blockEnd = 0;
        size_t idxInBlock = 0; // index into current decompressed block
        BlockData blk;
        bool eof = false;
    };

    bool openList(const string& term, ListIter& it) {
        cerr << "[INFO] Opening list for term: '" << term << "'\n";
        auto p = L_.find(term);
        if (p == L_.end()) {
            cerr << "[INFO] Term '" << term << "' not found in index\n";
            return false;
        }
        cerr << "[INFO] Term '" << term << "' found: " << p->second.postings << " postings, blocks " << p->second.startBlock << "-" << p->second.endBlock << "\n";
        it.lex = &p->second;
        it.curBlock = it.lex->startBlock;
        it.blockEnd = it.lex->endBlock;
        it.idxInBlock = 0;
        it.blk.docIDs.clear();
        it.blk.freqs.clear();
        it.eof = false;
        if (it.curBlock > it.blockEnd) { it.eof = true; return true; }
        // preload first block
        cerr << "[INFO] Loading first block " << it.curBlock << "...\n";
        loadBlock(it.curBlock, it.blk);
        cerr << "[INFO] First block loaded with " << it.blk.docIDs.size() << " postings\n";
        it.idxInBlock = 0;
        // Move idx to first posting of this term inside first block if lists share blocks:
        // (If your builder lets lists start/end mid-block, the lexicon should also carry
        // per-term first/last slot; for simplicity we assume blocks here fully belong to term.)
        return true;
    }

    // next posting (docID,freq). Returns false at end.
    bool next(ListIter& it, uint32_t& doc, uint32_t& freq) {
        if (it.eof) return false;
        while (true) {
            if (it.idxInBlock < it.blk.docIDs.size()) {
                doc = it.blk.docIDs[it.idxInBlock];
                freq= it.blk.freqs[it.idxInBlock];
                ++it.idxInBlock;
                return true;
            }
            // advance to next block
            if (it.curBlock >= it.blockEnd) {
                it.eof = true; return false;
            }
            ++it.curBlock;
            loadBlock(it.curBlock, it.blk);
            it.idxInBlock = 0;
        }
    }

    // Decode the block with global block index bIdx into bd
    void loadBlock(uint32_t bIdx, BlockData& bd) {
        static bool debug = getenv("DEBUG_QUERY") != nullptr;
        
        if (debug) cerr << "[DEBUG] loadBlock " << bIdx << "\n";
        
        uint64_t base = meta_.blockByteOff[bIdx];
        uint32_t dsz = meta_.docBytes[bIdx];
        uint32_t fsz = meta_.freqBytes[bIdx];

        if (debug) cerr << "[DEBUG]   base=" << base << " dsz=" << dsz << " fsz=" << fsz << "\n";

        // read doc bytes
        vector<uint8_t> buf(dsz + fsz);
        inv_.seekg((std::streamoff)base);
        inv_.read(reinterpret_cast<char*>(buf.data()), buf.size());
        const uint8_t* pDoc = buf.data();
        const uint8_t* pFreq= buf.data() + dsz;
        const uint8_t* endDoc= pFreq;
        const uint8_t* endFreq= buf.data() + buf.size();

        if (debug) cerr << "[DEBUG]   Decoding docs...\n";
        
        // decode doc gaps & reconstruct absolute docIDs
        bd.docIDs.clear(); bd.freqs.clear();
        // We don't know count; use freq stream as limiter. Strategy:
        // - Decode docs until we hit endDoc; decode freqs until endFreq; counts must match.
        vector<uint32_t> gaps;
        int doc_count = 0;
        while (pDoc < endDoc) {
            if (debug && doc_count % 10 == 0) cerr << "[DEBUG]     doc " << doc_count << " pDoc offset=" << (pDoc - buf.data()) << "\n";
            gaps.push_back(VarByte::decode(pDoc));
            doc_count++;
            if (doc_count > 10000) {
                cerr << "[ERROR] Infinite loop in doc decoding! Breaking.\n";
                break;
            }
        }
        
        if (debug) cerr << "[DEBUG]   Decoded " << gaps.size() << " doc gaps\n";
        if (debug) cerr << "[DEBUG]   Decoding freqs...\n";
        
        int freq_count = 0;
        while (pFreq < endFreq) {
            if (debug && freq_count % 10 == 0) cerr << "[DEBUG]     freq " << freq_count << " pFreq offset=" << (pFreq - buf.data()) << "\n";
            bd.freqs.push_back(VarByte::decode(pFreq));
            freq_count++;
            if (freq_count > 10000) {
                cerr << "[ERROR] Infinite loop in freq decoding! Breaking.\n";
                break;
            }
        }
        
        if (debug) cerr << "[DEBUG]   Decoded " << bd.freqs.size() << " freqs\n";
        
        if (gaps.size() != bd.freqs.size()) {
            if (debug) cerr << "[DEBUG]   Size mismatch! Trying lockstep decode...\n";
            // If mismatch: fall back to "try decode in lockstep" (safer if encoders did per-post var-length)
            bd.freqs.clear(); pDoc = buf.data(); pFreq = buf.data()+dsz;
            uint32_t last=0;
            bd.docIDs.clear(); bd.freqs.clear();
            int lockstep_count = 0;
            while (pDoc < endDoc && pFreq < endFreq) {
                uint32_t g = VarByte::decode(pDoc);
                uint32_t f = VarByte::decode(pFreq);
                last += g; bd.docIDs.push_back(last); bd.freqs.push_back(f);
                lockstep_count++;
                if (lockstep_count > 10000) {
                    cerr << "[ERROR] Infinite loop in lockstep decoding! Breaking.\n";
                    break;
                }
            }
            if (debug) cerr << "[DEBUG]   Lockstep decoded " << bd.docIDs.size() << " postings\n";
            return;
        }
        uint32_t last=0;
        for (size_t i=0;i<gaps.size();++i) { last += gaps[i]; bd.docIDs.push_back(last); }
        
        if (debug) cerr << "[DEBUG]   Block loaded with " << bd.docIDs.size() << " postings\n";
    }

    // term -> ft (postings)
    uint32_t ft(const string& term) const {
        auto it = L_.find(term);
        return it==L_.end() ? 0u : it->second.postings;
    }

    const unordered_map<string, LexiconEntry>& lexicon() const { return L_; }
    size_t numBlocks() const { return meta_.docBytes.size(); }

private:
    Config cfg_;
    Meta meta_;
    string invPath_;
    ifstream inv_;
    unordered_map<string, LexiconEntry> L_;
};

// ------------------------------- BM25 --------------------------------
struct BM25 {
    double k1, b, avgdl;
    uint32_t N;
    explicit BM25(double k1, double b, double avgdl, uint32_t N)
        : k1(k1), b(b), avgdl(avgdl>0?avgdl:1.0), N(N?N:1) {}

    static inline double idf(uint32_t N, uint32_t ft) {
        // classic BM25 idf
        double num = (double)N - (double)ft + 0.5;
        double den = (double)ft + 0.5;
        if (num <= 0) num = 1e-6;
        return log((num / den) + 1e-12);
    }

    inline double score(uint32_t ft, uint32_t fd, uint32_t doclen) const {
        double IDF = idf(N, ft);
        double K = k1 * (1.0 - b + b * (doclen / avgdl));
        return IDF * ((fd * (k1 + 1.0)) / (fd + K));
    }
};

// ------------------------------ Query Exec ---------------------------
struct QueryOptions {
    bool conjunctive = false;
    size_t topk = 10;
};

struct Hit { uint32_t doc; double score; };

static vector<Hit> runQuery(IndexReader& ir,
                            const vector<string>& terms,
                            const DocStats& dsOpt,
                            const QueryOptions& opt)
{
    // Build iterators per term
    vector<IndexReader::ListIter> I;
    I.reserve(terms.size());
    for (auto& t : terms) {
        IndexReader::ListIter it;
        if (!ir.openList(t, it)) continue; // term not in index
        I.push_back(std::move(it));
    }
    if (I.empty()) return {};

    // Prepare BM25 (N & avgdl). If collection.tsv missing, estimate N/avgdl from max docID seen lazily.
    DocStats ds = dsOpt;
    if (ds.N == 0) { ds.N = 1000000; ds.avgdl = 200.0; } // fallbacks

    BM25 bm25(CFG.k1, CFG.b, ds.avgdl, ds.N);

    // We’ll do a simple DAAT merge.
    // Strategy:
    //  - Disjunctive (OR): accumulate scores for any doc that appears in at least one list.
    //  - Conjunctive (AND): only keep docs that appear in all lists.
    // This implementation fully decompresses blocks on demand and advances pointers.

    // Read first postings
    struct Cur { uint32_t doc=UINT32_MAX; uint32_t tf=0; bool eof=true; };
    vector<Cur> cur(I.size());
    auto advance = [&](size_t i)->bool {
        uint32_t d,f;
        if (ir.next(I[i], d, f)) { cur[i] = {d,f,false}; return true; }
        cur[i] = {UINT32_MAX,0,true}; return false;
    };
    for (size_t i=0;i<I.size();++i) advance(i);

    auto allEof = [&](){
        for (auto &c: cur) if (!c.eof) return false; return true;
    };

    unordered_map<uint32_t,double> accum; accum.reserve(4096);

    // ft per term
    vector<uint32_t> fts; fts.reserve(terms.size());
    for (auto &t : terms) fts.push_back(ir.ft(t));

    // Helper to fetch doclen (fallback: sum of tfs seen so far for that doc)
    unordered_map<uint32_t,uint32_t> seenLen;

    while (!allEof()) {
        // find minimum docID among current
        uint32_t mind = UINT32_MAX;
        for (auto &c: cur) if (!c.eof) mind = min(mind, c.doc);

        // collect this doc's tfs across lists
        bool presentAll = true;
        vector<pair<size_t,uint32_t>> present; present.reserve(cur.size());
        for (size_t i=0;i<cur.size();++i) {
            if (!cur[i].eof && cur[i].doc == mind) {
                present.push_back({i, cur[i].tf});
            } else {
                if (opt.conjunctive) presentAll = false;
            }
        }
        if (!opt.conjunctive || presentAll) {
            // Document length:
            uint32_t dlen = 0;
            if (ds.len.size() > mind) dlen = ds.len[mind];
            if (dlen == 0) {
                // fall back to quick proxy: running sum of tfs we've seen for this doc
                uint32_t sumtf=0; for (auto &pr: present) sumtf += pr.second;
                dlen = max(1u, seenLen[mind] += sumtf);
            }
            // score sum
            double s = 0.0;
            for (size_t j=0;j<present.size();++j) {
                size_t i = present[j].first;
                uint32_t tf = present[j].second;
                s += bm25.score(fts[i], tf, dlen);
            }
            if (s != 0.0) accum[mind] += s;
        }

        // advance lists that matched mind
        for (size_t i=0;i<cur.size();++i) {
            if (!cur[i].eof && cur[i].doc == mind) advance(i);
            else if (opt.conjunctive && !presentAll) {
                // AND-mode: we must raise lower docs to mind
                // naive catch-up: advance until >= mind or eof
                while (!cur[i].eof && cur[i].doc < mind) {
                    if (!advance(i)) break;
                }
            }
        }
    }

    // build top-k
    vector<Hit> hits; hits.reserve(accum.size());
    for (auto &kv : accum) hits.push_back({kv.first, kv.second});
    partial_sort(hits.begin(), hits.begin()+min(hits.size(), CFG.topk), hits.end(),
                 [](const Hit& a, const Hit& b){ return a.score > b.score; });
    if (hits.size() > CFG.topk) hits.resize(CFG.topk);
    return hits;
}

// ------------------------------ CLI parsing --------------------------
static void usage(const char* argv0) {
    cerr <<
    "Usage:\n"
    "  " << argv0 << " [-i index_dir] [-k topk] [--and|--or] [-v] query terms...\n"
    "Options:\n"
    "  -i DIR     index directory (default: index_output)\n"
    "  -k K       top-K results (default: 10)\n"
    "  --and      conjunctive mode (AND)\n"
    "  --or       disjunctive mode (OR, default)\n"
    "  -v         verbose mode (show detailed logs)\n"
    "Examples:\n"
    "  " << argv0 << " pizza hut\n"
    "  " << argv0 << " --and new york university\n"
    "  " << argv0 << " -v -k 20 search engine\n";
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // parse args
    vector<string> terms;
    bool verbose = false;
    for (int i=1;i<argc;i++) {
        string a = argv[i];
        if (a=="-i" && i+1<argc) { CFG.index_dir = argv[++i]; }
        else if (a=="-k" && i+1<argc) { CFG.topk = (size_t)stoull(argv[++i]); }
        else if (a=="--and") { CFG.conjunctive = true; }
        else if (a=="--or")  { CFG.conjunctive = false; }
        else if (a=="-v" || a=="--verbose") { verbose = true; }
        else if (!a.empty() && a[0]=='-') { usage(argv[0]); return 1; }
        else { terms.push_back(a); }
    }
    if (terms.empty()) {
        usage(argv[0]);
        return 0;
    }

    try {
        if (verbose) {
            cerr << "🔍 Query Processing Started\n";
            cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
            cerr << "📝 Query terms: ";
            for (size_t i=0;i<terms.size();++i){ if(i) cerr<<", "; cerr<<"'"<<terms[i]<<"'"; }
            cerr << "\n";
            cerr << "⚙️  Mode: " << (CFG.conjunctive ? "AND (conjunctive)" : "OR (disjunctive)") << "\n";
            cerr << "📊 Top-K: " << CFG.topk << "\n";
            cerr << "📁 Index dir: " << CFG.index_dir << "\n";
            cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
        }

        // Load optional doc stats (N, avgdl, lengths)
        if (verbose) cerr << "📖 Loading document statistics...\n";
        DocStats ds = loadDocStatsOptional(pathJoin(CFG.index_dir, CFG.collection_tsv));
        if (verbose) {
            cerr << "   • Documents: " << ds.N << "\n";
            cerr << "   • Average doc length: " << ds.avgdl << "\n\n";
        }

        // Open index
        if (verbose) cerr << "📂 Loading index metadata...\n";
        IndexReader ir(CFG);
        if (verbose) {
            cerr << "   • Blocks loaded: " << ir.numBlocks() << "\n";
            cerr << "   • Terms in lexicon: " << ir.lexicon().size() << "\n\n";
        }

        // Check which terms are in index
        if (verbose) {
            cerr << "🔎 Checking term frequencies:\n";
            for (const auto& t : terms) {
                uint32_t ft = ir.ft(t);
                if (ft > 0) {
                    cerr << "   • '" << t << "': " << ft << " documents\n";
                } else {
                    cerr << "   • '" << t << "': NOT FOUND in index\n";
                }
            }
            cerr << "\n";
        }

        // Query
        if (verbose) cerr << "🚀 Executing query...\n";
        QueryOptions opt;
        opt.conjunctive = CFG.conjunctive;
        opt.topk = CFG.topk;

        auto start_time = std::chrono::high_resolution_clock::now();
        auto hits = runQuery(ir, terms, ds, opt);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if (verbose) {
            cerr << "   ✓ Query completed in " << duration.count() << "ms\n";
            cerr << "   ✓ Found " << hits.size() << " results\n\n";
            cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        }

        // Print results
        cout.setf(std::ios::fixed); cout<<setprecision(6);
        cout << "Mode: " << (opt.conjunctive ? "AND" : "OR") << "\n";
        cout << "Top " << hits.size() << " results for: ";
        for (size_t i=0;i<terms.size();++i){ if(i) cout<<' '; cout<<terms[i]; }
        cout << "\n-----------------------------------------\n";
        for (size_t i=0;i<hits.size();++i) {
            cout << setw(2) << (i+1) << ". docID=" << hits[i].doc
                 << "  score=" << hits[i].score << "\n";
        }
        if (hits.empty()) cout << "(no results)\n";
        
        if (verbose) {
            cout << "\n⏱️  Query time: " << duration.count() << "ms\n";
        }
    } catch (const exception& e) {
        cerr << "❌ ERROR: " << e.what() << "\n";
        return 2;
    }
    return 0;
}