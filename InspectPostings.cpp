#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

#pragma pack(push, 1)
struct RunHeader {
    uint32_t magic;  // 'POST' = 0x504F5354
    uint16_t ver;    // 1
    uint16_t le;     // 1 if little-endian host wrote this
};
#pragma pack(pop)

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <postings.tmp> [max_records]\n";
        return 1;
    }
    const char* path = argv[1];
    uint64_t max_records = (argc > 2) ? std::stoull(argv[2]) : 50;

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
        if (!in.read(term.data(), static_cast<std::streamsize>(term_len))) break;

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