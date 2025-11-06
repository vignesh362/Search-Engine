// BuildPassageStore.cpp
// Build: clang++ -std=c++17 -O2 BuildPassageStore.cpp -o BuildPassageStore
//
// PURPOSE:
//   Build offsets.bin and texts.bin for SnippetExtractor from collection.tsv
//
// USAGE:
//   ./BuildPassageStore collection.tsv index_output
//
// OUTPUT:
//   Creates index_output/offsets.bin and index_output/texts.bin

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>
#include <sys/stat.h>

using namespace std;

bool directoryExists(const string& path) {
    struct stat info;
    return (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <collection.tsv> <output_dir>\n";
        cerr << "Example: " << argv[0] << " collection.tsv index_output\n";
        return 1;
    }

    string inputFile = argv[1];
    string outputDir = argv[2];

    // Check if output directory exists
    if (!directoryExists(outputDir)) {
        cerr << "Error: Output directory '" << outputDir << "' does not exist\n";
        return 1;
    }

    string offsetsPath = outputDir + "/offsets.bin";
    string textsPath = outputDir + "/texts.bin";

    ifstream input(inputFile);
    if (!input) {
        cerr << "Error: Cannot open input file '" << inputFile << "'\n";
        return 1;
    }

    ofstream textsOut(textsPath, ios::binary);
    if (!textsOut) {
        cerr << "Error: Cannot create '" << textsPath << "'\n";
        return 1;
    }

    vector<uint64_t> offsets;
    uint64_t currentOffset = 0;
    uint32_t maxDocID = 0;
    size_t lineCount = 0;
    vector<uint32_t> docLengths; // token counts per docID

    cout << "Building passage store from " << inputFile << "...\n";

    string line;
    while (getline(input, line)) {
        lineCount++;
        if (line.empty()) continue;

        // Parse: docID\ttext
        size_t tabPos = line.find('\t');
        if (tabPos == string::npos) {
            cerr << "Warning: Line " << lineCount << " has no tab, skipping\n";
            continue;
        }

        uint32_t docID = stoul(line.substr(0, tabPos));
        string text = line.substr(tabPos + 1);

        // Ensure offsets vector is large enough
        if (docID >= offsets.size()) {
            offsets.resize(docID + 1, 0);
            docLengths.resize(docID + 1, 0);
        }

        // Store offset for this document
        offsets[docID] = currentOffset;
        
        // Write text to texts.bin
        textsOut.write(text.c_str(), text.size());
        currentOffset += text.size();

        // Compute token count (simple whitespace split, skip empty tokens)
        uint32_t tokCount = 0;
        {
            std::istringstream iss(text);
            std::string tok;
            while (iss >> tok) {
                // basic cleaning: keep tokens that have at least one alnum
                bool hasAlnum = false;
                for (unsigned char c : tok) if (std::isalnum(c)) { hasAlnum = true; break; }
                if (hasAlnum) ++tokCount;
            }
        }

        // Ensure docLengths vector is large enough (in case offsets resized earlier didn't)
        if (docID >= docLengths.size()) docLengths.resize(docID + 1, 0);
        docLengths[docID] = tokCount;

        if (docID > maxDocID) maxDocID = docID;

        if (lineCount % 100000 == 0) {
            cout << "  Processed " << lineCount << " documents...\n";
        }
    }

    // Add sentinel offset at the end
    offsets.push_back(currentOffset);
    // Add sentinel doc length (0) for end
    docLengths.push_back(0);

    cout << "  Total documents processed: " << lineCount << "\n";
    cout << "  Max docID: " << maxDocID << "\n";
    cout << "  Total text size: " << currentOffset << " bytes\n";

    // Close texts.bin
    textsOut.close();

    // Write offsets.bin
    ofstream offsetsOut(offsetsPath, ios::binary);
    if (!offsetsOut) {
        cerr << "Error: Cannot create '" << offsetsPath << "'\n";
        return 1;
    }

    offsetsOut.write(reinterpret_cast<const char*>(offsets.data()), 
                     offsets.size() * sizeof(uint64_t));
    offsetsOut.close();

    // Write docstats.bin (uint32_t token counts per docID)
    string docstatsPath = outputDir + "/docstats.bin";
    ofstream docstatsOut(docstatsPath, ios::binary);
    if (!docstatsOut) {
        cerr << "Warning: Cannot create '" << docstatsPath << "' - skipping docstats\n";
    } else {
        docstatsOut.write(reinterpret_cast<const char*>(docLengths.data()),
                          docLengths.size() * sizeof(uint32_t));
        docstatsOut.close();
        cout << "  Wrote docstats: " << docstatsPath << " (" << (docLengths.size() * sizeof(uint32_t)) << " bytes)\n";
    }

    cout << "\nSuccessfully created:\n";
    cout << "  " << offsetsPath << " (" << (offsets.size() * sizeof(uint64_t)) << " bytes)\n";
    cout << "  " << textsPath << " (" << currentOffset << " bytes)\n";
    cout << "\nPassage store is ready for SnippetExtractor!\n";

    return 0;
}

