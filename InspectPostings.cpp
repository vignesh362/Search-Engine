#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

#pragma pack(push, 1)
struct RunHeader {
    uint32_t magic;  // 'POST' = 0x504F5354
    uint16_t ver;    // 1
    uint16_t le;     // 1 if little-endian host wrote this
};
#pragma pack(pop)

// VarByte decoder
static inline uint64_t vb_decode_uint(std::ifstream& in) {
    uint64_t val = 0;
    uint8_t byte;
    // First byte has MSB=0
    in.read(reinterpret_cast<char*>(&byte), 1);
    val = byte & 0x7F;
    // Continue bytes have MSB=1
    while (in && (in.peek() & 0x80)) {
        in.read(reinterpret_cast<char*>(&byte), 1);
        val = (val << 7) | (byte & 0x7F);
    }
    return val;
}

struct LexEntry {
    std::string term;
    uint64_t start_slot;
    uint64_t end_slot;
    uint32_t first_block;
    uint32_t num_blocks;
    uint32_t ft;
};

static size_t get_file_size(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return static_cast<size_t>(f.tellg());
}

static std::vector<uint32_t> read_u32_array(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + p.string());
    f.seekg(0, std::ios::end);
    size_t bytes = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> v(bytes / sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    return v;
}

static std::vector<LexEntry> read_lexicon(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("Cannot open " + p.string());
    std::vector<LexEntry> entries;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        LexEntry e;
        ss >> e.term >> e.start_slot >> e.end_slot >> e.first_block >> e.num_blocks >> e.ft;
        entries.push_back(e);
    }
    return entries;
}

static bool selfcheck(const fs::path& index_dir) {
    std::cout << "Running self-check on index: " << index_dir << "\n";
    std::cout << "================================================\n";
    
    try {
        // 1. Load metadata arrays
        auto docidsize = read_u32_array(index_dir / "docidsize.bin");
        auto freqsize = read_u32_array(index_dir / "freqsize.bin");
        auto lastdocid = read_u32_array(index_dir / "lastdocid.bin");
        auto lexicon = read_lexicon(index_dir / "lexicon.tsv");
        size_t invlist_size = get_file_size(index_dir / "invlists.bin");
        
        std::cout << "Loaded metadata:\n";
        std::cout << "  docidsize entries: " << docidsize.size() << "\n";
        std::cout << "  freqsize entries: " << freqsize.size() << "\n";
        std::cout << "  lastdocid entries: " << lastdocid.size() << "\n";
        std::cout << "  lexicon entries: " << lexicon.size() << "\n";
        std::cout << "  invlists.bin size: " << invlist_size << " bytes\n\n";
        
        // 2. Verify array lengths match
        if (docidsize.size() != freqsize.size() || docidsize.size() != lastdocid.size()) {
            std::cout << "FAIL: Metadata array lengths don't match!\n";
            std::cout << "  docidsize: " << docidsize.size() 
                      << ", freqsize: " << freqsize.size() 
                      << ", lastdocid: " << lastdocid.size() << "\n";
            return false;
        }
        std::cout << "✓ All metadata arrays have equal length (" << docidsize.size() << ")\n";
        
        // 3. Verify sum of sizes matches invlists.bin size
        uint64_t total_bytes = 0;
        for (size_t i = 0; i < docidsize.size(); ++i) {
            total_bytes += docidsize[i] + freqsize[i];
        }
        if (total_bytes != invlist_size) {
            std::cout << "FAIL: Sum of block sizes doesn't match invlists.bin size!\n";
            std::cout << "  Σ(docidsize+freqsize) = " << total_bytes << "\n";
            std::cout << "  invlists.bin size = " << invlist_size << "\n";
            return false;
        }
        std::cout << "✓ Σ(docidsize+freqsize) == invlists.bin size (" << total_bytes << " bytes)\n\n";
        
        // 4. Sample terms and verify postings
        std::cout << "Sampling terms for posting verification...\n";
        const size_t sample_size = std::min<size_t>(100, lexicon.size());
        
        // Create sample: mix of random + high-frequency terms
        std::vector<size_t> sample_indices;
        std::mt19937 rng(42);
        
        // Sort by ft to get top terms
        std::vector<size_t> sorted_indices(lexicon.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), 
                  [&](size_t a, size_t b) { return lexicon[a].ft > lexicon[b].ft; });
        
        // Take top 50 and 50 random
        for (size_t i = 0; i < std::min<size_t>(50, lexicon.size()); ++i) {
            sample_indices.push_back(sorted_indices[i]);
        }
        std::shuffle(sorted_indices.begin(), sorted_indices.end(), rng);
        for (size_t i = 0; i < std::min<size_t>(50, lexicon.size()); ++i) {
            if (sample_indices.size() >= sample_size) break;
            sample_indices.push_back(sorted_indices[i]);
        }
        
        std::ifstream invlist(index_dir / "invlists.bin", std::ios::binary);
        if (!invlist) {
            std::cout << "FAIL: Cannot open invlists.bin\n";
            return false;
        }
        
        size_t checked = 0;
        for (size_t idx : sample_indices) {
            const auto& entry = lexicon[idx];
            
            // Calculate byte offset for first block
            uint64_t byte_offset = 0;
            for (uint32_t b = 0; b < entry.first_block; ++b) {
                byte_offset += docidsize[b] + freqsize[b];
            }
            
            // Read all blocks for this term
            std::vector<uint32_t> docids;
            
            for (uint32_t b = 0; b < entry.num_blocks; ++b) {
                uint32_t block_idx = entry.first_block + b;
                if (block_idx >= docidsize.size()) {
                    std::cout << "FAIL: term '" << entry.term << "' block " << b 
                              << " index out of range (" << block_idx << " >= " << docidsize.size() << ")\n";
                    return false;
                }
                
                // Seek to this block
                invlist.seekg(static_cast<std::streamoff>(byte_offset));
                
                // Decode docids from this block
                uint32_t did_bytes = docidsize[block_idx];
                uint32_t frq_bytes = freqsize[block_idx];
                
                // Read docid deltas
                std::streamoff did_end = invlist.tellg() + static_cast<std::streamoff>(did_bytes);
                std::vector<uint32_t> block_dids;
                uint32_t prev = 0;
                while (invlist.tellg() < did_end) {
                    uint64_t delta = vb_decode_uint(invlist);
                    if (block_dids.empty()) {
                        prev = static_cast<uint32_t>(delta); // first is absolute
                    } else {
                        prev += static_cast<uint32_t>(delta); // rest are gaps
                    }
                    block_dids.push_back(prev);
                }
                
                // Verify strict increasing order within block
                for (size_t i = 1; i < block_dids.size(); ++i) {
                    if (block_dids[i] <= block_dids[i-1]) {
                        std::cout << "FAIL: term '" << entry.term << "' block " << b 
                                  << " docIDs not strictly increasing: "
                                  << block_dids[i-1] << " >= " << block_dids[i] << "\n";
                        return false;
                    }
                }
                
                // Verify increasing across blocks
                if (!docids.empty() && !block_dids.empty()) {
                    if (block_dids[0] <= docids.back()) {
                        std::cout << "FAIL: term '" << entry.term << "' block " << b 
                                  << " first docID (" << block_dids[0] 
                                  << ") not greater than previous block last (" << docids.back() << ")\n";
                        return false;
                    }
                }
                
                // Verify last docid matches metadata
                if (!block_dids.empty() && block_dids.back() != lastdocid[block_idx]) {
                    std::cout << "FAIL: term '" << entry.term << "' block " << b 
                              << " last docID (" << block_dids.back() 
                              << ") doesn't match metadata (" << lastdocid[block_idx] << ")\n";
                    return false;
                }
                
                docids.insert(docids.end(), block_dids.begin(), block_dids.end());
                byte_offset += did_bytes + frq_bytes;
            }
            
            // Verify total count matches ft
            if (docids.size() != entry.ft) {
                std::cout << "FAIL: term '" << entry.term << "' decoded " << docids.size() 
                          << " postings but ft=" << entry.ft << "\n";
                return false;
            }
            
            ++checked;
            if (checked % 10 == 0) {
                std::cout << "  Checked " << checked << "/" << sample_indices.size() << " terms...\r" << std::flush;
            }
        }
        
        std::cout << "✓ Checked " << checked << " terms: all have strict docID order and count == ft\n";
        
        std::cout << "\n================================================\n";
        std::cout << "PASS: All checks successful!\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "FAIL: Exception during self-check: " << e.what() << "\n";
        return false;
    }
}

static int inspect_postings(const char* path, uint64_t max_records) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open: " << path << "\n";
        return 1;
    }

    // Get file size
    in.seekg(0, std::ios::end);
    std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    // Read and validate header
    RunHeader h{};
    if (!in.read(reinterpret_cast<char*>(&h), sizeof(h))) {
        std::cerr << "File too small for header\n";
        return 1;
    }
    if (h.magic != 0x504F5354u) {
        std::cerr << "Bad magic; not a postings run file\n";
        return 1;
    }

    // Data region ends before 8-byte footer (record count)
    std::streamoff data_end = file_size >= static_cast<std::streamoff>(sizeof(RunHeader) + sizeof(uint64_t))
        ? file_size - static_cast<std::streamoff>(sizeof(uint64_t))
        : static_cast<std::streamoff>(sizeof(RunHeader));

    uint64_t count = 0;
    while (in && count < max_records && in.tellg() < data_end) {
        uint32_t term_len = 0;
        if (!in.read(reinterpret_cast<char*>(&term_len), sizeof(term_len))) break;

        if (in.tellg() + static_cast<std::streamoff>(term_len + sizeof(uint64_t) + sizeof(uint32_t)) > data_end) {
            break; // would overrun data region
        }

        std::string term(term_len, '\0');
        if (!in.read(const_cast<char*>(term.data()), static_cast<std::streamsize>(term_len))) break;

        uint64_t docID = 0;
        uint32_t freq = 0;
        if (!in.read(reinterpret_cast<char*>(&docID), sizeof(docID))) break;
        if (!in.read(reinterpret_cast<char*>(&freq), sizeof(freq))) break;

        std::cout << term << "\t" << docID << "\t" << freq << "\n";
        ++count;
    }

    // Optionally read footer record count
    in.clear();
    in.seekg(-static_cast<std::streamoff>(sizeof(uint64_t)), std::ios::end);
    uint64_t footer_count = 0;
    if (in.read(reinterpret_cast<char*>(&footer_count), sizeof(footer_count))) {
        std::cerr << "Read records: " << count << " (footer says " << footer_count << " total)\n";
    } else {
        std::cerr << "Read records: " << count << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <postings.tmp> [max_records]  - inspect temp postings\n"
                  << "  " << argv[0] << " --selfcheck <index_dir>       - validate final index\n";
        return 1;
    }
    
    std::string arg1 = argv[1];
    if (arg1 == "--selfcheck") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " --selfcheck <index_dir>\n";
            return 1;
        }
        return selfcheck(argv[2]) ? 0 : 1;
    }
    
    // Default: inspect postings
    uint64_t max_records = (argc > 2) ? std::stoull(argv[2]) : 50;
    return inspect_postings(argv[1], max_records);
}