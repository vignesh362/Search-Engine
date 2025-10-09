// BuildIndex.cpp
// Merge -> Block -> VarByte compress
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace fs = std::filesystem;

// ---------- VarByte (VB) codec ----------
static inline void vb_encode_uint(uint64_t x, std::string& out) {
    // write 7 bits per byte, last byte MSB=0, others MSB=1
    uint8_t buf[10]; // enough for 64-bit
    int n = 0;
    do {
        buf[n++] = static_cast<uint8_t>(x & 0x7F);
        x >>= 7;
    } while (x != 0);
    // last byte first (MSB=0), then set MSB=1 for all but last
    out.push_back(static_cast<char>(buf[n - 1])); // MSB=0
    for (int i = n - 2; i >= 0; --i) {
        out.push_back(static_cast<char>(buf[i] | 0x80)); // MSB=1
    }
}
static inline void vb_encode_seq_u32(const std::vector<uint32_t>& v, std::string& out) {
    for (uint32_t x : v) vb_encode_uint(x, out);
}

// ---------- Run reader (reads your intermediate records) ----------
struct Rec {
    std::string term;
    uint64_t docID;
    uint32_t freq;
    int run_id; // which run this came from
};

struct RunReader {
    std::ifstream in;
    uint64_t records_remaining = 0;
    bool header_read = false;
    
    explicit RunReader(const fs::path& p) : in(p, std::ios::binary) {
        if (!in) throw std::runtime_error("Failed to open run: " + p.string());
        // Optional: set a big read buffer
        static constexpr size_t BUF = 1 << 20;
        auto* buf = new char[BUF];
        in.rdbuf()->pubsetbuf(buf, BUF);
    }
    bool next(Rec& out) {
        // Read header if not done yet
        if (!header_read) {
            uint32_t magic;
            uint16_t ver, le;
            if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic))) return false;
            if (!in.read(reinterpret_cast<char*>(&ver), sizeof(ver))) return false;
            if (!in.read(reinterpret_cast<char*>(&le), sizeof(le))) return false;
            
            // Validate magic number (0x504F5354 = 'POST')
            if (magic != 0x504F5354u) {
                throw std::runtime_error("Invalid magic number in posting file");
            }
            
            // Read record count (written after header)
            if (!in.read(reinterpret_cast<char*>(&records_remaining), sizeof(records_remaining))) return false;
            header_read = true;
        }
        
        // Try to read record
        uint32_t term_len = 0;
        if (!in.read(reinterpret_cast<char*>(&term_len), sizeof(term_len))) {
            // End of data section, try to read footer
            if (!in.read(reinterpret_cast<char*>(&records_remaining), sizeof(records_remaining))) {
                return false; // End of file
            }
            return false; // No more records
        }
        
        std::string term(term_len, '\0');
        if (!in.read(const_cast<char*>(term.data()), static_cast<std::streamsize>(term_len))) return false;
        uint64_t did = 0;
        uint32_t fr = 0;
        if (!in.read(reinterpret_cast<char*>(&did), sizeof(did))) return false;
        if (!in.read(reinterpret_cast<char*>(&fr), sizeof(fr))) return false;
        
        out.term = std::move(term);
        out.docID = did;
        out.freq  = fr;
        return true;
    }
};

struct RecLess {
    bool operator()(const Rec& a, const Rec& b) const {
        if (a.term != b.term) return a.term > b.term;   // min-heap via '>'
        if (a.docID != b.docID) return a.docID > b.docID;
        return a.freq > b.freq; // tie-breaker (not used functionally)
    }
};

// ---------- Lexicon row ----------
struct LexRow {
    uint64_t start_slot = 0;    // global posting slot start (inclusive)
    uint64_t end_slot   = 0;    // global posting slot end (exclusive)
    uint32_t first_block = 0;   // index into metadata arrays
    uint32_t num_blocks  = 0;
    uint32_t ft          = 0;   // document frequency (#postings)
};

// ---------- Block flush ----------
static void flush_block(const std::vector<uint32_t>& dids,
                        const std::vector<uint32_t>& frqs,
                        std::ofstream& inv,
                        std::vector<uint32_t>& lastdocid,
                        std::vector<uint32_t>& docidsize,
                        std::vector<uint32_t>& freqsize) {
    // Build deltas for docIDs: [abs first, gaps...]
    std::vector<uint32_t> deltas;
    deltas.reserve(dids.size());
    deltas.push_back(dids[0]);
    for (size_t i = 1; i < dids.size(); ++i) {
        uint32_t gap = dids[i] - dids[i - 1];
        deltas.push_back(gap);
    }

    // Encode
    std::string buf_doc, buf_frq;
    buf_doc.reserve(deltas.size() * 2);
    buf_frq.reserve(frqs.size() * 2);
    vb_encode_seq_u32(deltas, buf_doc);
    vb_encode_seq_u32(frqs,  buf_frq);

    // Write [doc_block][freq_block]
    inv.write(buf_doc.data(), static_cast<std::streamsize>(buf_doc.size()));
    inv.write(buf_frq.data(), static_cast<std::streamsize>(buf_frq.size()));

    // Metadata
    lastdocid.push_back(dids.back());
    docidsize.push_back(static_cast<uint32_t>(buf_doc.size()));
    freqsize.push_back(static_cast<uint32_t>(buf_frq.size()));
}

// ---------- Main ----------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr <<
            "Usage:\n"
            "  " << argv[0] << " <tmp_runs_dir> <out_dir> [block_size]\n\n"
            "Inputs:\n  <tmp_runs_dir>  directory containing postings_*.tmp\n"
            "Outputs in <out_dir>:\n"
            "  invlists.bin, lastdocid.bin, docidsize.bin, freqsize.bin, lexicon.tsv\n";
        return 1;
    }
    fs::path runs_dir = argv[1];
    fs::path out_dir  = argv[2];
    const size_t B    = (argc > 3) ? static_cast<size_t>(std::stoull(argv[3])) : 128;

    if (!fs::exists(runs_dir) || !fs::is_directory(runs_dir)) {
        std::cerr << "Runs dir not found: " << runs_dir << "\n";
        return 1;
    }
    fs::create_directories(out_dir);

    // Collect run files
    std::vector<fs::path> run_paths;
    for (auto& ent : fs::directory_iterator(runs_dir)) {
        if (!ent.is_regular_file()) continue;
        auto name = ent.path().filename().string();
        if (name.rfind("postings_", 0) == 0 && ent.path().extension() == ".tmp")
            run_paths.push_back(ent.path());
    }
    if (run_paths.empty()) {
        std::cerr << "No postings_*.tmp found in " << runs_dir << "\n";
        return 1;
    }
    std::sort(run_paths.begin(), run_paths.end()); // nice deterministic order

    // Open run readers
    std::vector<std::unique_ptr<RunReader>> readers;
    readers.reserve(run_paths.size());
    for (auto& p : run_paths) {
        try {
            readers.emplace_back(std::make_unique<RunReader>(p));
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }

    // Output files
    std::ofstream inv(out_dir / "invlists.bin", std::ios::binary);
    if (!inv) { std::cerr << "Cannot open invlists.bin for writing\n"; return 1; }
    // Large write buffer
    static constexpr size_t WBUF = 1 << 20;
    auto* wbuf = new char[WBUF];
    inv.rdbuf()->pubsetbuf(wbuf, WBUF);

    std::vector<uint32_t> lastdocid, docidsize, freqsize;
    lastdocid.reserve(1 << 20);
    docidsize.reserve(1 << 20);
    freqsize.reserve(1 << 20);

    // Min-heap priming
    std::priority_queue<Rec, std::vector<Rec>, RecLess> pq;
    for (int i = 0; i < static_cast<int>(readers.size()); ++i) {
        Rec r; r.run_id = i;
        if (readers[i]->next(r)) pq.push(std::move(r));
    }
    if (pq.empty()) {
        std::cerr << "All runs empty\n";
        return 1;
    }

    // Merge state
    std::unordered_map<std::string, LexRow> lex; // store in memory then write TSV
    lex.reserve(1 << 20);

    std::string cur_term;
    uint64_t last_doc = 0;
    uint32_t acc_freq = 0;

    std::vector<uint32_t> dids; dids.reserve(B);
    std::vector<uint32_t> frqs; frqs.reserve(B);

    uint64_t global_slot = 0;
    uint32_t next_block_index = 0;
    LexRow cur;

    auto start_term = [&](const std::string& t) {
        cur_term = t;
        last_doc = 0; acc_freq = 0;
        dids.clear(); frqs.clear();
        cur = {};
        cur.start_slot  = global_slot;
        cur.first_block = next_block_index;
        cur.ft          = 0;
    };
    auto flush_current_doc_if_any = [&]() {
        if (acc_freq > 0) {
            dids.push_back(static_cast<uint32_t>(last_doc));
            frqs.push_back(acc_freq);
            ++global_slot;
            ++cur.ft;
            acc_freq = 0;
            if (dids.size() == B) {
                flush_block(dids, frqs, inv, lastdocid, docidsize, freqsize);
                dids.clear(); frqs.clear();
                ++next_block_index;
            }
        }
    };
    auto finish_term = [&]() {
        flush_current_doc_if_any();
        if (!dids.empty()) {
            flush_block(dids, frqs, inv, lastdocid, docidsize, freqsize);
            dids.clear(); frqs.clear();
            ++next_block_index;
        }
        cur.num_blocks = next_block_index - cur.first_block;
        cur.end_slot   = global_slot; // exclusive
        lex.emplace(cur_term, cur);
        cur_term.clear();
    };

    // Main merge
    while (!pq.empty()) {
        Rec r = pq.top(); pq.pop();
        if (cur_term.empty()) start_term(r.term);
        if (r.term != cur_term) {
            finish_term();
            start_term(r.term);
        }
        // same term: combine same docID
        if (last_doc == r.docID) {
            acc_freq += r.freq;
        } else {
            flush_current_doc_if_any();
            last_doc = r.docID;
            acc_freq = r.freq;
        }

        // read next from the same run
        Rec nxt; nxt.run_id = r.run_id;
        if (readers[r.run_id]->next(nxt)) pq.push(std::move(nxt));
    }
    if (!cur_term.empty()) finish_term();

    // Write metadata arrays
    auto write_vec_u32 = [&](const fs::path& p, const std::vector<uint32_t>& v) {
        std::ofstream f(p, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot write " + p.string());
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(uint32_t)));
    };
    write_vec_u32(out_dir / "lastdocid.bin", lastdocid);
    write_vec_u32(out_dir / "docidsize.bin", docidsize);
    write_vec_u32(out_dir / "freqsize.bin",  freqsize);

    // Write lexicon TSV
    std::ofstream lexf(out_dir / "lexicon.tsv");
    if (!lexf) { std::cerr << "Cannot write lexicon.tsv\n"; return 1; }
    lexf << "term\tstart_slot\tend_slot\tfirst_block\tnum_blocks\tft\n";
    for (const auto& kv : lex) {
        const auto& t  = kv.first;
        const auto& lr = kv.second;
        lexf << t << '\t'
             << lr.start_slot  << '\t'
             << lr.end_slot    << '\t'
             << lr.first_block << '\t'
             << lr.num_blocks  << '\t'
             << lr.ft          << '\n';
    }

    std::cout << "OK. Blocks: " << next_block_index
              << "  Terms: " << lex.size()
              << "  invlists.bin bytes written.\n";
    return 0;
}