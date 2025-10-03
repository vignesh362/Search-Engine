#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;

struct Posting {
    std::string term;
    uint64_t docID;
    uint32_t freq;
};

static void clean_term(std::string& s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) {
        return !std::isalpha(c);
    }), s.end());
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
}

static void flush_buffer_to_temp(const std::vector<Posting>& buffer, const fs::path& temp_dir, size_t file_index) {
    fs::create_directories(temp_dir);
    fs::path out_path = temp_dir / (std::string("postings_") + std::to_string(file_index) + ".tmp");
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open temp file for writing: " << out_path << std::endl;
        return;
    }
    for (const Posting& p : buffer) {
        uint32_t term_len = static_cast<uint32_t>(p.term.size());
        out.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
        out.write(p.term.data(), static_cast<std::streamsize>(p.term.size()));
        out.write(reinterpret_cast<const char*>(&p.docID), sizeof(p.docID));
        out.write(reinterpret_cast<const char*>(&p.freq), sizeof(p.freq));
    }
}

int main(int argc, char** argv) {
    fs::path input_file = argc > 1 ? fs::path(argv[1]) : fs::path("collection.tsv");
    fs::path temp_dir = argc > 2 ? fs::path(argv[2]) : fs::path("tmp_postings");
    size_t buffer_capacity = argc > 3 ? static_cast<size_t>(std::stoull(argv[3])) : static_cast<size_t>(200'000);
    uint64_t max_docs = argc > 4 ? static_cast<uint64_t>(std::stoull(argv[4])) : static_cast<uint64_t>(100);

    if (!fs::exists(input_file)) {
        std::cerr << "Input file does not exist: " << input_file << std::endl;
        return 1;
    }

    std::ifstream in(input_file);
    if (!in) {
        std::cerr << "Failed to open input file: " << input_file << std::endl;
        return 1;
    }

    std::vector<Posting> buffer;
    buffer.reserve(buffer_capacity);
    size_t temp_index = 0;
    uint64_t processed_docs = 0;
    std::string line;

    while (processed_docs < max_docs && std::getline(in, line)) {
        if (line.empty()) { ++processed_docs; continue; }
        // MS MARCO style: docid \t text
        size_t tab = line.find('\t');
        std::string text = (tab == std::string::npos) ? line : line.substr(tab + 1);

        std::stringstream ss(text);
        std::string token;
        std::unordered_map<std::string, uint32_t> term_freq;
        while (ss >> token) {
            clean_term(token);
            if (!token.empty()) {
                ++term_freq[token];
            }
        }
        uint64_t docID = processed_docs + 1; // sequential for testing
        for (const auto& kv : term_freq) {
            buffer.push_back(Posting{kv.first, docID, kv.second});
        }
        if (buffer.size() >= buffer_capacity) {
            std::sort(buffer.begin(), buffer.end(), [](const Posting& a, const Posting& b){
                if (a.term != b.term) return a.term < b.term;
                if (a.docID != b.docID) return a.docID < b.docID;
                return a.freq < b.freq;
            });
            flush_buffer_to_temp(buffer, temp_dir, temp_index++);
            buffer.clear();
        }
        ++processed_docs;
    }

    if (!buffer.empty()) {
        std::sort(buffer.begin(), buffer.end(), [](const Posting& a, const Posting& b){
            if (a.term != b.term) return a.term < b.term;
            if (a.docID != b.docID) return a.docID < b.docID;
            return a.freq < b.freq;
        });
        flush_buffer_to_temp(buffer, temp_dir, temp_index++);
        buffer.clear();
    }

    std::cout << "Processed documents: " << processed_docs << std::endl;
    std::cout << "Intermediate postings written to: " << temp_dir << std::endl;
    return 0;
}