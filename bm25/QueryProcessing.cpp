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
    bool load_docstats = true; // toggle loading/processing of collection.tsv for doc lengths
    bool verbose = false; // enable verbose runtime logging
} CFG;

// ------------------------- Small utils / IO ---------------------------
static inline bool fileExists(const string& p) {
    ifstream in(p, ios::binary);
    return in.good();
}

// (removed unused split_ws helper)

static inline string join(const vector<string>& v, const string& sep) {
    ostringstream oss;
    for (size_t i=0;i<v.size();++i) {
        if (i) oss << sep;
        oss << v[i];
    }
    return oss.str();
}

// ------------------------------- VarByte ------------------------------
// VarByte format from BuildIndex: buf[n-1] first (< 128), then buf[n-2..0] with MSB=1 (>= 128)
// So: [high_7bits < 128], [mid_7bits | 0x80], ..., [low_7bits | 0x80]
// Example: 128 = [1, 128]: (1 << 7) | (128 & 0x7F) = 128 + 0 = 128
// Example: 300 = [2, 172]: (2 << 7) | (172 & 0x7F) = 256 + 44 = 300
struct VarByte {
    // decode one uint32 from ptr (advance ptr)
    static inline uint32_t decode(const uint8_t *&p, const uint8_t *end) {
        // First byte always has MSB=0 (value 0-127)
        uint8_t b = *p++;
        uint32_t x = b;
        
        // Continue reading while we see bytes with MSB=1 (>= 128)
        // These are the continuation bytes
        while (p < end && *p >= 128) {
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
        for (int i = n - 2; i >= 0; --i)
            out.push_back((uint8_t)(buf[i] | 0x80));
    }
};

// ------------------------- Binary vector loaders ----------------------
static vector<uint32_t> loadU32Bin(const string& filePath) {
    ifstream in(filePath, ios::binary);
    if (!in) throw runtime_error("Cannot open file: " + filePath);
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
    uint64_t offset     = 0; // byte offset into invlists.bin (unused here)
};

// Header-aware loader: expects columns: term, first_block, num_blocks, ft (others allowed)
static unordered_map<string, LexiconEntry> loadLexicon(const string& filePath) {
    ifstream in(filePath);
    if (!in) throw runtime_error("Cannot open lexicon: " + filePath);

    string header;
    if (!getline(in, header)) throw runtime_error("Empty lexicon: " + filePath);

    // Split header on tabs
    vector<string> cols;
    {
        string t;
        for (char c: header) {
            if (c=='\t') { cols.push_back(t); t.clear(); }
            else t.push_back(c);
        }
        cols.push_back(t);
    }

    auto findCol = [&](const string& name)->int{
        for (int i=0;i<(int)cols.size();++i) if (cols[i]==name) return i;
        return -1;
    };

    int c_term        = findCol("term");
    int c_first_block = findCol("first_block");
    int c_num_blocks  = findCol("num_blocks");
    int c_ft          = findCol("ft");

    if (c_term<0 || c_first_block<0 || c_num_blocks<0 || c_ft<0) {
        throw runtime_error("lexicon.tsv missing required columns: term/first_block/num_blocks/ft");
    }

    unordered_map<string, LexiconEntry> L;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;

        vector<string> tok;
        {
            string t;
            for (char c: line) {
                if (c=='\t') { tok.push_back(t); t.clear(); }
                else t.push_back(c);
            }
            tok.push_back(t);
        }
        if ((int)tok.size() <= max({c_term, c_first_block, c_num_blocks, c_ft})) continue;

        LexiconEntry e;
        e.term = tok[c_term];
        uint32_t fb = (uint32_t)stoul(tok[c_first_block]);
        uint32_t nb = (uint32_t)stoul(tok[c_num_blocks]);
        e.startBlock = fb;
        e.endBlock   = nb ? (fb + nb - 1) : fb;
        e.postings   = (uint32_t)stoul(tok[c_ft]);
        e.offset     = 0;

        L.emplace(e.term, e);
    }
    if (L.empty()) throw runtime_error("Lexicon parsed 0 entries. Check format/columns.");
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
    // Allow caller to disable expensive stats pass (useful for large collections)
    if (!CFG.load_docstats) return ds;
    // Prefer a precomputed binary docstats if available (fast)
    string docstats_bin = CFG.index_dir + "/docstats.bin";
    if (fileExists(docstats_bin)) {
        try {
            vector<uint32_t> lens = loadU32Bin(docstats_bin);
            uint64_t tot = 0;
            for (uint32_t v : lens) tot += v;
            ds.len = std::move(lens);
            ds.N = (uint32_t)ds.len.size();
            ds.totalLen = tot;
            ds.avgdl = ds.N ? double(tot)/double(ds.N) : 1.0;
            cerr << "  [docstats] Loaded precomputed docstats (" << ds.N << " docs)\n";
            return ds;
        } catch (const exception& ex) {
            cerr << "  [docstats] Failed to load docstats.bin: " << ex.what() << "\n";
            // fall through to scanning collection
        }
    }

    if (!fileExists(collection_tsv)) return ds;
    ifstream in(collection_tsv);
    if (!in) return ds;
    string line;
    uint64_t tot=0;
    vector<uint32_t> lens;

    // Progress reporting
    const uint64_t REPORT_INTERVAL = 100000; // lines
    uint64_t lines = 0;
    auto t0 = chrono::steady_clock::now();
    while (getline(in, line)) {
        ++lines;
        if (line.empty()) continue;

        // Parse as: external_id \t text  (MSMARCO style). If no tab, whole line is text
        string text;
        size_t tabPos = line.find('\t');
        if (tabPos == string::npos) text = line;
        else text = line.substr(tabPos + 1);

        // Count tokens by whitespace and simple alnum check
        uint32_t tokCount = 0;
        {
            istringstream iss(text);
            string tok;
            while (iss >> tok) {
                bool hasAlnum = false;
                for (unsigned char c : tok) if (isalnum(c)) { hasAlnum = true; break; }
                if (hasAlnum) ++tokCount;
            }
        }

        lens.push_back(tokCount);
        tot += tokCount;

        if ((lines % REPORT_INTERVAL) == 0) {
            auto t1 = chrono::steady_clock::now();
            double secs = chrono::duration<double>(t1 - t0).count();
            double rate = secs > 0 ? (double)lines / secs : 0.0;
            cerr << "  [docstats] Processed " << lines << " lines (" << (uint64_t)rate << "/s)\n";
        }
    }
    ds.len = std::move(lens);
    ds.N = (uint32_t)ds.len.size();
    ds.totalLen = tot;
    ds.avgdl = ds.N ? double(tot)/double(ds.N) : 1.0;
    return ds;
}

// ---------------------------- Index Reader ---------------------------
struct BlockData {
    vector<uint32_t> docIDs;
    vector<uint32_t> freqs;
};

struct IndexReader {
    ifstream inv_;
    vector<uint32_t> lastDocId;  // last docID of each block (delta-encoded base)
    vector<uint32_t> docBytes;   // bytes for doc deltas per block
    vector<uint32_t> freqBytes;  // bytes for freq VB per block
    vector<uint64_t> blockByteOff;// starting byte offsets per block (prefix sum of (docBytes+freqBytes))

    IndexReader(const string& dir) {
        string inv = dir + "/" + CFG.invlists_bin;
        string last= dir + "/" + CFG.lastdocid_bin;
        string dsz = dir + "/" + CFG.docidsize_bin;
        string fsz = dir + "/" + CFG.freqsize_bin;

        if (!fileExists(inv) || !fileExists(last) || !fileExists(dsz) || !fileExists(fsz)) {
            throw runtime_error("Index files missing in " + dir);
        }

        inv_.open(inv, ios::binary);
        if (!inv_) throw runtime_error("Cannot open invlists.bin");

        lastDocId = loadU32Bin(last);
        docBytes  = loadU32Bin(dsz);
        freqBytes = loadU32Bin(fsz);
        if (lastDocId.size()!=docBytes.size() || docBytes.size()!=freqBytes.size())
            throw runtime_error("Metadata vectors have different sizes");

        blockByteOff.resize(docBytes.size());
        uint64_t off=0;
        for (size_t i=0;i<docBytes.size();++i) {
            blockByteOff[i] = off;
            off += (uint64_t)docBytes[i] + (uint64_t)freqBytes[i];
        }
    }

    // Decode the block with global block index bIdx into bd
    void loadBlock(uint32_t bIdx, BlockData& bd) {
        if (bIdx >= docBytes.size()) {
            throw runtime_error(string("Block index out of range: ") + to_string(bIdx) +
                                " (blocks=" + to_string(docBytes.size()) + ")");
        }
        static bool debug = getenv("DEBUG_QUERY") != nullptr;
        
        if (debug) cerr << "[DEBUG] loadBlock " << bIdx << "\n";
        
        uint64_t base = blockByteOff[bIdx];
        uint32_t dsz = docBytes[bIdx];
        uint32_t fsz = freqBytes[bIdx];

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

        // doc deltas: first is base (absolute lastDocId[bIdx]), followed by gaps backward? or forward?
        // Assumption: BuildIndex wrote per-block as:
        //   - for docIDs: VB-encoded GAPS (forward) from previous docID, with first gap from 0 produces first docID
        //   - lastDocId[bIdx] is *last* docID of the block (sanity usage)
        // Here we simply decode gaps and accumulate.
        bd.docIDs.clear(); bd.freqs.clear();
        bd.docIDs.reserve(256);
        bd.freqs.reserve(256);

        uint32_t prev = 0;
        while (pDoc < endDoc) {
            uint32_t gap = VarByte::decode(pDoc, endDoc);
            uint32_t id  = prev + gap;
            bd.docIDs.push_back(id);
            prev = id;
        }

        if (debug) cerr << "[DEBUG]   Decoding freqs...\n";

        while (pFreq < endFreq) {
            uint32_t f = VarByte::decode(pFreq, endFreq);
            bd.freqs.push_back(f);
        }

        if (bd.docIDs.size()!=bd.freqs.size()) {
            throw runtime_error("Decoded block docIDs and freqs size mismatch: " +
                                to_string(bd.docIDs.size()) + " vs " + to_string(bd.freqs.size()));
        }
    }
};

// ------------------------------ Scoring -------------------------------
static inline double idf_BM25(uint32_t N, uint32_t ft) {
    // BM25 IDF with +0.5 smoothing
    return log(( (double)N - (double)ft + 0.5 ) / ( (double)ft + 0.5 ) + 1.0);
}

static inline double tf_BM25(uint32_t tf, uint32_t dl, double avgdl) {
    double k1 = CFG.k1, b = CFG.b;
    double denom = tf + k1 * (1 - b + b * (double(dl) / avgdl));
    return (tf * (k1 + 1.0)) / (denom > 0 ? denom : 1.0);
}

// ------------------------------ Iterator with Block Skipping ------------------------------
struct PostingsIter {  // Enhanced Block-by-Block Iterator with Block Skipping
    // term scope
    uint32_t blockStart = 0;
    uint32_t blockEnd   = 0;
    uint32_t curBlock   = 0;
    bool eof = true;

    BlockData blk;
    size_t idxInBlock = 0;

    // block store
    IndexReader* rdr = nullptr;

    bool init(IndexReader* r, uint32_t bStart, uint32_t bEnd) {
        rdr = r; blockStart=bStart; blockEnd=bEnd; curBlock=bStart;
        eof = false;
        if (curBlock > blockEnd) { eof = true; return true; }
        // preload first block
        cerr << "[INFO] Loading first block " << curBlock << "...\n";
        rdr->loadBlock(curBlock, blk);
        cerr << "[INFO] First block loaded with " << blk.docIDs.size() << " postings\n";
        idxInBlock = 0;
        return true;
    }

    bool has() const {
        return !eof && (idxInBlock < blk.docIDs.size());
    }

    uint32_t doc() const {
        return blk.docIDs[idxInBlock];
    }

    uint32_t tf() const {
        return blk.freqs[idxInBlock];
    }

    bool next() {
        if (eof) return false;
        ++idxInBlock;
        while (idxInBlock >= blk.docIDs.size()) {
            // move to next block
            if (curBlock >= blockEnd) {
                eof = true; return false;
            }
            ++curBlock;
            rdr->loadBlock(curBlock, blk);
            idxInBlock = 0;
        }
        return true;
    }

    // Block skipping functionality
    bool skipToDoc(uint32_t targetDocId) {
        if (eof) return false;
        
        // Skip blocks that can't contain the target document
        while (curBlock < blockEnd && rdr->lastDocId[curBlock] < targetDocId) {
            ++curBlock;
        }
        
        if (curBlock >= blockEnd) {
            eof = true;
            return false;
        }
        
        // Load the block that might contain our target
        rdr->loadBlock(curBlock, blk);
        idxInBlock = 0;
        
        // If this block's first doc is still greater than target, we're done
        if (!blk.docIDs.empty() && blk.docIDs[0] > targetDocId) {
            eof = true;
            return false;
        }
        
        // Find the position within the block
        while (idxInBlock < blk.docIDs.size() && blk.docIDs[idxInBlock] < targetDocId) {
            ++idxInBlock;
        }
        
        return idxInBlock < blk.docIDs.size();
    }
};

// ------------------------------- Query -------------------------------
struct QueryEnv {
    unordered_map<string, LexiconEntry> lex;
    IndexReader* rdr = nullptr;
    DocStats ds;
};

static void banner(const vector<string>& terms, const string& mode) {
    cerr << "🔍 Query Processing Started (with Block Skipping)\n";
    cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    cerr << "📝 Query terms: '" << join(terms, " ") << "'\n";
    cerr << "⚙️  Mode: " << mode << "\n";
    cerr << "📊 Top-K: " << CFG.topk << "\n";
    cerr << "📁 Index dir: " << CFG.index_dir << "\n";
    cerr << "🚀 Block Skipping: ENABLED\n";
    cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
}

static QueryEnv setup(const vector<string>& terms) {
    QueryEnv env;

    // Doc stats (optional)
    cerr << "📖 Loading document statistics...\n";
    env.ds = loadDocStatsOptional(CFG.index_dir + "/" + CFG.collection_tsv);
    cerr << "   • Documents: " << env.ds.N << "\n";
    cerr << "   • Average doc length: " << (uint64_t)round(env.ds.avgdl) << "\n\n";

    // Reader
    env.rdr = new IndexReader(CFG.index_dir);

    // Lexicon
    cerr << "📂 Loading index metadata...\n";
    env.lex = loadLexicon(CFG.index_dir + "/" + CFG.lexicon_tsv);
    cerr << "   • Blocks loaded: " << env.rdr->docBytes.size() << "\n";
    cerr << "   • Terms in lexicon: " << env.lex.size() << "\n\n";

    // Show ft info
    cerr << "🔎 Checking term frequencies:\n";
    for (auto& t: terms) {
        auto it = env.lex.find(t);
        if (it==env.lex.end()) {
            cerr << "   • '"<<t<<"': (not in lexicon)\n";
        } else {
            cerr << "   • '"<<t<<"': " << it->second.postings << " documents\n";
        }
    }
    cerr << "\n";
    return env;
}

// Multi-term query processing with DAAT traversal and block skipping
struct TermIterator {
    string term;
    LexiconEntry entry;
    PostingsIter iter;
    uint32_t currentDoc;
    uint32_t currentTf;
    double idf;
    bool valid;
    
    TermIterator(const string& t, const LexiconEntry& e, IndexReader* rdr, uint32_t N) 
        : term(t), entry(e), currentDoc(0), currentTf(0), valid(false) {
        iter.init(rdr, e.startBlock, e.endBlock);
        idf = idf_BM25(N, e.postings);
        advance();
    }
    
    void advance() {
        if (iter.has()) {
            currentDoc = iter.doc();
            currentTf = iter.tf();
            valid = true;
        } else {
            valid = false;
        }
    }
    
    void next() {
        iter.next();
        advance();
    }
    
    // NEW: Skip to a specific document using block skipping
    bool skipToDoc(uint32_t targetDocId) {
        if (iter.skipToDoc(targetDocId)) {
            advance();
            return valid;
        }
        valid = false;
        return false;
    }
    
    bool has() const { return valid; }
    uint32_t doc() const { return currentDoc; }
    uint32_t tf() const { return currentTf; }
};

// Enhanced multi-term OR processing with block skipping
static void runMultiTerm_OR(const vector<string>& terms, QueryEnv& env) {
    cerr << "🚀 Executing multi-term OR query with block skipping...\n";
    
    vector<TermIterator> iterators;
    uint32_t N = env.ds.N ? env.ds.N : 1u<<31;
    
    // Initialize iterators for all terms
    for (const auto& term : terms) {
        auto it = env.lex.find(term);
        if (it == env.lex.end()) {
            cerr << "[WARN] Term '"<<term<<"' not found in lexicon.\n";
            continue;
        }
        
        cerr << "[INFO] Term '"<<term<<"' found: " << it->second.postings << " postings\n";
        iterators.emplace_back(term, it->second, env.rdr, N);
    }
    
    if (iterators.empty()) {
        cerr << "[WARN] No valid terms found.\n";
        return;
    }
    
    size_t K = CFG.topk;
    vector<pair<double,uint32_t>> top; // score, docID
    top.reserve(K+1);
    
    double avgdl = env.ds.avgdl > 0 ? env.ds.avgdl : 1.0;
    unordered_map<uint32_t, double> docScores; // docID -> total score
    
    // DAAT traversal with block skipping
    while (true) {
        uint32_t minDoc = UINT32_MAX;
        
        // Find minimum doc ID among all iterators
        for (auto& it : iterators) {
            if (it.has() && it.doc() < minDoc) {
                minDoc = it.doc();
            }
        }
        
        if (minDoc == UINT32_MAX) break; // All iterators exhausted
        
        // NEW: Skip iterators that are behind minDoc using block skipping
        for (auto& it : iterators) {
            if (it.has() && it.doc() < minDoc) {
                it.skipToDoc(minDoc);
            }
        }
        
        // Accumulate scores for current document
        double docScore = 0.0;
        uint32_t dl = (env.ds.len.size() > minDoc ? env.ds.len[minDoc] : 1u);
        
        for (auto& it : iterators) {
            if (it.has() && it.doc() == minDoc) {
                double termScore = it.idf * tf_BM25(it.tf(), dl, avgdl);
                docScore += termScore;
                it.next(); // Advance this iterator
            }
        }
        
        docScores[minDoc] = docScore;
        
        // Update top-K heap
        if (top.size() < K) {
            top.emplace_back(docScore, minDoc);
            if (top.size() == K) {
                nth_element(top.begin(), top.begin()+K-1, top.end(),
                            [](auto&a, auto&b){ return a.first > b.first; });
            }
        } else if (docScore > top[K-1].first) {
            top[K-1] = {docScore, minDoc};
            nth_element(top.begin(), top.begin()+K-1, top.end(),
                        [](auto&a, auto&b){ return a.first > b.first; });
        }
    }
    
    sort(top.begin(), top.end(), [](auto&a, auto&b){ return a.first > b.first; });
    cout << "Top " << top.size() << " results for OR query (with block skipping):\n";
    for (size_t i=0;i<top.size();++i) {
        cout << setw(2) << (i+1) << ". doc=" << top[i].second << " score=" << fixed << setprecision(4) << top[i].first << "\n";
    }
}

// OPTIMIZED multi-term AND processing with efficient early termination
static void runMultiTerm_AND(const vector<string>& terms, QueryEnv& env) {
    cerr << "🚀 Executing OPTIMIZED multi-term AND query with early termination...\n";
    
    vector<TermIterator> iterators;
    uint32_t N = env.ds.N ? env.ds.N : 1u<<31;
    
    // Initialize iterators for all terms
    for (const auto& term : terms) {
        auto it = env.lex.find(term);
        if (it == env.lex.end()) {
            cerr << "[WARN] Term '"<<term<<"' not found in lexicon.\n";
            cerr << "[INFO] AND query requires ALL terms to be found.\n";
            return;
        }
        
        cerr << "[INFO] Term '"<<term<<"' found: " << it->second.postings << " postings\n";
        iterators.emplace_back(term, it->second, env.rdr, N);
    }
    
    if (iterators.empty()) {
        cerr << "[WARN] No valid terms found.\n";
        return;
    }
    
    size_t K = CFG.topk;
    vector<pair<double,uint32_t>> top; // score, docID
    top.reserve(K+1);
    
    double avgdl = env.ds.avgdl > 0 ? env.ds.avgdl : 1.0;
    
    // OPTIMIZED DAAT traversal for AND with efficient early termination
    while (true) {
        // Find minimum doc ID among all iterators
        uint32_t minDoc = UINT32_MAX;
        bool allValid = true;
        
        for (const auto& it : iterators) {
            if (!it.has()) {
                allValid = false;
                break;  // Early termination if ANY iterator exhausted
            }
            if (it.doc() < minDoc) {
                minDoc = it.doc();
            }
        }
        
        if (!allValid || minDoc == UINT32_MAX) break;
        
        // OPTIMIZATION: Quick check if all iterators have this document BEFORE expensive operations
        bool allHaveMinDoc = true;
        for (const auto& it : iterators) {
            if (it.doc() != minDoc) {
                allHaveMinDoc = false;
                break;  // Early exit - no need to process this document
            }
        }
        
        if (allHaveMinDoc) {
            // This document contains ALL terms - calculate score efficiently
            double docScore = 0.0;
            uint32_t dl = (env.ds.len.size() > minDoc ? env.ds.len[minDoc] : 1u);
            
            for (const auto& it : iterators) {
                double termScore = it.idf * tf_BM25(it.tf(), dl, avgdl);
                docScore += termScore;
            }
            
            // Update top-K heap
            if (top.size() < K) {
                top.emplace_back(docScore, minDoc);
                if (top.size() == K) {
                    nth_element(top.begin(), top.begin()+K-1, top.end(),
                                [](auto&a, auto&b){ return a.first > b.first; });
                }
            } else if (docScore > top[K-1].first) {
                top[K-1] = {docScore, minDoc};
                nth_element(top.begin(), top.begin()+K-1, top.end(),
                            [](auto&a, auto&b){ return a.first > b.first; });
            }
        }
        
        // Advance all iterators that are at minDoc
        for (auto& it : iterators) {
            if (it.has() && it.doc() == minDoc) {
                it.next();
            }
        }
    }
    
    sort(top.begin(), top.end(), [](auto&a, auto&b){ return a.first > b.first; });
    cout << "Top " << top.size() << " results for OPTIMIZED AND query:\n";
    for (size_t i=0;i<top.size();++i) {
        cout << setw(2) << (i+1) << ". doc=" << top[i].second << " score=" << fixed << setprecision(4) << top[i].first << "\n";
    }
}

// Simple BM25 disjunctive (OR) over single term
static void runSingleTerm_OR(const string& term, QueryEnv& env) {
    cerr << "🚀 Executing single-term query...\n";
    auto it = env.lex.find(term);
    if (it == env.lex.end()) {
        cerr << "[WARN] Term '"<<term<<"' not found in lexicon.\n";
        return;
    }

    auto e = it->second;
    cerr << "[INFO] Opening list for term: '"<<term<<"'\n";
    cerr << "[INFO] Term '"<<term<<"' found: " << e.postings << " postings, blocks "
         << e.startBlock << "-" << e.endBlock << "\n";

    PostingsIter pit;
    pit.init(env.rdr, e.startBlock, e.endBlock);

    size_t K = CFG.topk;
    vector<pair<double,uint32_t>> top; // score, docID
    top.reserve(K+1);

    uint32_t N = env.ds.N ? env.ds.N : 1u<<31;         // fallback
    double   avgdl = env.ds.avgdl > 0 ? env.ds.avgdl : 1.0;
    double   idf = idf_BM25(N, e.postings);

    while (pit.has()) {
        uint32_t d = pit.doc();
        uint32_t tf = pit.tf();
        // if we don't have doc lengths, fallback to 1
        uint32_t dl = (env.ds.len.size()>d ? env.ds.len[d] : 1u);

        double s = idf * tf_BM25(tf, dl, avgdl);

        if (top.size() < K) {
            top.emplace_back(s, d);
            if (top.size()==K) {
                nth_element(top.begin(), top.begin()+K-1, top.end(),
                            [](auto&a, auto&b){ return a.first>b.first; });
            }
        } else if (s > top[K-1].first) {
            top[K-1] = {s,d};
            nth_element(top.begin(), top.begin()+K-1, top.end(),
                        [](auto&a, auto&b){ return a.first>b.first; });
        }

        pit.next();
    }

    sort(top.begin(), top.end(), [](auto&a, auto&b){ return a.first>b.first; });
    cout << "Top " << top.size() << " results for '"<<term<<"':\n";
    for (size_t i=0;i<top.size();++i) {
        cout << setw(2) << (i+1) << ". doc=" << top[i].second << " score=" << fixed << setprecision(4) << top[i].first << "\n";
    }
}

// ------------------------------- main --------------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);

    vector<string> terms;
    bool verbose = false;
    for (int i=1;i<argc;++i) {
        string a = argv[i];
        if (a=="-k" && i+1<argc) {
            CFG.topk = (size_t)stoul(argv[++i]);
        } else if (a=="-and") {
            CFG.conjunctive = true;
        } else if (a=="--no-docstats") {
            CFG.load_docstats = false;
        } else if (a=="-v") {
            verbose = true;
            if (i+1<argc && argv[i+1][0] != '-') {
                terms.push_back(argv[++i]);
            }
        } else {
            terms.push_back(a);
        }
    }

    if (terms.empty()) {
        cerr << "Usage: " << argv[0] << " [-k K] [-and] term1 [term2 ...]\n";
        return 1;
    }

    banner(terms, CFG.conjunctive ? "AND (conjunctive)" : "OR (disjunctive)");

    try {
        auto env = setup(terms);

        if (terms.size() == 1) {
            // Single term query
            runSingleTerm_OR(terms[0], env);
        } else {
            // Multi-term query
            if (CFG.conjunctive) {
                runMultiTerm_AND(terms, env);
            } else {
                runMultiTerm_OR(terms, env);
            }
        }
    } catch (const exception& ex) {
        cerr << "[FATAL] " << ex.what() << "\n";
        return 2;
    }
    return 0;
}