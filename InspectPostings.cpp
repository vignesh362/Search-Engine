#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

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

    uint64_t count = 0;
    while (in && count < max_records) {
        uint32_t term_len = 0;
        if (!in.read(reinterpret_cast<char*>(&term_len), sizeof(term_len))) break;

        std::string term(term_len, '\0');
        if (!in.read(term.data(), static_cast<std::streamsize>(term_len))) break;

        uint64_t docID = 0;
        uint32_t freq = 0;
        if (!in.read(reinterpret_cast<char*>(&docID), sizeof(docID))) break;
        if (!in.read(reinterpret_cast<char*>(&freq), sizeof(freq))) break;

        std::cout << term << "\t" << docID << "\t" << freq << "\n";
        ++count;
    }
    std::cerr << "Read records: " << count << "\n";
    return 0;
}


