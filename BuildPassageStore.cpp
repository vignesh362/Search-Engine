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
        }

        // Store offset for this document
        offsets[docID] = currentOffset;
        
        // Write text to texts.bin
        textsOut.write(text.c_str(), text.size());
        currentOffset += text.size();

        if (docID > maxDocID) maxDocID = docID;

        if (lineCount % 100000 == 0) {
            cout << "  Processed " << lineCount << " documents...\n";
        }
    }

    // Add sentinel offset at the end
    offsets.push_back(currentOffset);

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

    cout << "\nSuccessfully created:\n";
    cout << "  " << offsetsPath << " (" << (offsets.size() * sizeof(uint64_t)) << " bytes)\n";
    cout << "  " << textsPath << " (" << currentOffset << " bytes)\n";
    cout << "\nPassage store is ready for SnippetExtractor!\n";

    return 0;
}

