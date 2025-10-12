// build_runs.cpp
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct Posting {
    std::string term;
    uint64_t docID;
    uint32_t freq;
};

// ---- term cleaning: keep letters & digits; lowercase ----
static inline void clean_term(std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalpha(c) || std::isdigit(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    s.swap(out);
}

// ---- tiny file header/footer for each temp run ----
#pragma pack(push, 1)
struct RunHeader {
    uint32_t magic;  // 'POST' = 0x504F5354
    uint16_t ver;    // 1
    uint16_t le;     // 1 if little-endian host wrote this
};
#pragma pack(pop)

static inline bool is_little_endian() {
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t*>(&x) == 1;
}

// zero-pad file index for nicer sorting on disk
static std::string zero_pad(size_t n, int width = 6) {
    std::string s = std::to_string(n);
    if (static_cast<int>(s.size()) < width) {
        s.insert(s.begin(), width - static_cast<int>(s.size()), '0');
    }
    return s;
}

// write one temp run: header, postings, footer(count)
static void flush_buffer_to_temp(const std::vector<Posting>& buffer,
                                 const fs::path& temp_dir,
                                 size_t file_index) {
    if (buffer.empty()) return;

    fs::create_directories(temp_dir);
    fs::path out_path = temp_dir / (std::string("postings_") + zero_pad(file_index) + ".tmp");
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open temp file for writing: " << out_path << "\n";
        return;
    }

    // Bigger write buffer to reduce syscalls
    static constexpr size_t IO_BUF_SIZE = 1 << 20; // 1 MiB
    std::vector<char> io_buf(IO_BUF_SIZE);
    out.rdbuf()->pubsetbuf(io_buf.data(), static_cast<std::streamsize>(io_buf.size()));

    // Header
    RunHeader h{0x504F5354u, 1u, static_cast<uint16_t>(is_little_endian() ? 1 : 0)};
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));

    // Record count (written after header)
    uint64_t records_written = buffer.size();
    out.write(reinterpret_cast<const char*>(&records_written), sizeof(records_written));

    // Records
    for (const Posting& p : buffer) {
        uint32_t term_len = static_cast<uint32_t>(p.term.size());
        out.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
        out.write(p.term.data(), static_cast<std::streamsize>(p.term.size()));
        out.write(reinterpret_cast<const char*>(&p.docID), sizeof(p.docID));
        out.write(reinterpret_cast<const char*>(&p.freq), sizeof(p.freq));
    }
    out.close();

    std::cout << "Wrote run #" << file_index
              << " (" << records_written << " records) → " << out_path << "\n";
}

static inline size_t approx_posting_bytes(const Posting& p) {
    // layout: 4 (len) + term bytes + 8 (docID) + 4 (freq)
    return 16 + p.term.size();
}

// Usage:
//   build_runs <collection.tsv> [temp_dir] [flush_mb] [max_docs]
// Defaults:
//   temp_dir = tmp_postings
//   flush_mb = 128
//   max_docs = UINT64_MAX (process entire collection)
int main(int argc, char** argv) {
    fs::path input_file = argc > 1 ? fs::path(argv[1]) : fs::path("collection.tsv");
    fs::path temp_dir   = argc > 2 ? fs::path(argv[2]) : fs::path("tmp_postings");
    size_t flush_mb     = argc > 3 ? static_cast<size_t>(std::stoull(argv[3])) : static_cast<size_t>(128);
    uint64_t max_docs   = argc > 4 ? static_cast<uint64_t>(std::stoull(argv[4])) : UINT64_MAX;

    if (!fs::exists(input_file)) {
        std::cerr << "Input file does not exist: " << input_file << "\n";
        return 1;
    }

    std::ifstream in(input_file);
    if (!in) {
        std::cerr << "Failed to open input file: " << input_file << "\n";
        return 1;
    }

    fs::create_directories(temp_dir);
    // Sidecar: map docID -> external id (left field before tab if present; else line number)
    std::ofstream docmap_out(temp_dir / "docmap.tsv", std::ios::app);
    if (!docmap_out) {
        std::cerr << "Failed to open docmap.tsv for writing in " << temp_dir << "\n";
        return 1;
    }

    std::vector<Posting> buffer;
    buffer.reserve(200'000); // reserve some; actual flush controlled by byte cap

    const size_t byte_cap = flush_mb * 1024ull * 1024ull;
    size_t approx_bytes_in_buffer = 0;

    size_t temp_index = 0;
    uint64_t processed_docs = 0;
    std::string line;

    while (processed_docs < max_docs && std::getline(in, line)) {
        if (line.empty()) { ++processed_docs; continue; }

        // Parse "external_id \t text" (MS MARCO style); fall back if no tab
        std::string external_id, text;
        if (size_t tab = line.find('\t'); tab != std::string::npos) {
            external_id = line.substr(0, tab);
            text = line.substr(tab + 1);
        } else {
            external_id.clear();
            text = line;
        }

        // Tokenize on whitespace, clean each token, count per-doc frequencies
        std::unordered_map<std::string, uint32_t> term_freq;
        term_freq.reserve(256);
        std::stringstream ss(text);
        std::string token;
        while (ss >> token) {
            clean_term(token);
            if (!token.empty()) ++term_freq[token];
        }

        // Use external_id as docID, convert to uint64_t
        uint64_t docID;
        if (external_id.empty()) {
            // If no external_id, use sequential number as fallback
            docID = processed_docs + 1;
            external_id = std::to_string(docID);
        } else {
            // Convert external_id string to uint64_t
            try {
                docID = std::stoull(external_id);
            } catch (const std::exception&) {
                // If conversion fails, use sequential number as fallback
                docID = processed_docs + 1;
                std::cerr << "Warning: Could not convert external_id '" << external_id 
                         << "' to number, using fallback docID " << docID << std::endl;
            }
        }
        // record docID mapping (now docID == external_id)
        docmap_out << docID << '\t' << external_id << '\n';

        // stage postings
        for (const auto& kv : term_freq) {
            Posting p{kv.first, docID, kv.second};
            approx_bytes_in_buffer += approx_posting_bytes(p);
            buffer.emplace_back(std::move(p));

            // Flush if we exceed approx byte cap (and keep a floor to avoid tiny runs)
            if (approx_bytes_in_buffer >= byte_cap && buffer.size() >= 1000) {
                // Sort by (term, docID) ONLY
                std::sort(buffer.begin(), buffer.end(),
                          [](const Posting& a, const Posting& b) {
                              if (a.term != b.term) return a.term < b.term;
                              return a.docID < b.docID;
                          });
                flush_buffer_to_temp(buffer, temp_dir, temp_index++);
                buffer.clear();
                approx_bytes_in_buffer = 0;
            }
        }

        ++processed_docs;
    }

    // Final flush
    if (!buffer.empty()) {
        std::sort(buffer.begin(), buffer.end(),
                  [](const Posting& a, const Posting& b) {
                      if (a.term != b.term) return a.term < b.term;
                      return a.docID < b.docID;
                  });
        flush_buffer_to_temp(buffer, temp_dir, temp_index++);
        buffer.clear();
    }

    std::cout << "Processed documents: " << processed_docs << "\n";
    std::cout << "Intermediate runs written to: " << temp_dir << "\n";
    std::cout << "Doc map sidecar: " << (temp_dir / "docmap.tsv") << "\n";
    return 0;
}