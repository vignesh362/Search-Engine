// SnippetExtractor.cpp
// Build: clang++ -std=c++17 -O2 SnippetExtractor.cpp -o SnippetExtractor
//
// PURPOSE
//   Given a list of top docIDs and the query terms, produce query-dependent snippets
//   by loading the original passages from a compact store and selecting best-scoring
//   windows (coverage + proximity + position) with highlighted terms.
//
// PASSAGE STORE FORMAT (simple, fast, file-based):
//   store_dir/
//     offsets.bin : uint64_t array of size (N + 1)
//                   offsets[i] = byte start of doc i in texts.bin
//                   offsets[N] = byte end of last doc (sentinel)
//     texts.bin   : UTF-8 bytes; for each doc i, bytes [offsets[i], offsets[i+1]) is the text
//
// INPUT (CLI):
//   ./SnippetExtractor --store-dir index_output --docs 12,418,777 --query "university ranking research"
//   OR pass docIDs via a file:
//   ./SnippetExtractor --store-dir index_output --docs-file top_docs.txt --query "university ranking research"
//
// OUTPUT:
//   JSON lines to stdout; each document emits:
//   {
//     "docID": 12,
//     "snippets": ["...first highlighted snippet...", "...second highlighted snippet..."]
//   }
//
// NOTES:
//   • Window size is token-based (default 40 tokens).
//   • Scoring favors windows that cover more unique query terms, higher total matches,
//     tighter proximity (shorter span), and a small early-position bonus.
//   • Highlights wrap exact word matches (case-insensitive) with <b>…</b>.
//
//   If you don’t have offsets/bin yet, see the helper instructions at the end.

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <cstdint>
#include <cctype>
using namespace std;

struct Args {
    string storeDir = "index_output";
    vector<uint32_t> docIDs;
    string docsFile;
    string query;
    size_t windowsz = 40;  // tokens per window
    size_t perDoc = 2;     // snippets per doc
};

static void usage(const char* prog) {
    cerr <<
    "Usage:\n"
    "  " << prog << " --store-dir DIR --query \"q terms\" --docs 1,2,3 [--windowsz 40] [--per 2]\n"
    "  " << prog << " --store-dir DIR --query \"q terms\" --docs-file top_docs.txt [--windowsz 40] [--per 2]\n";
}

static bool parseCSVDocIDs(const string& s, vector<uint32_t>& out) {
    if (s.empty()) return false;
    string num; out.clear();
    for (char c: s) {
        if (c==',' || c==' ' || c=='\t') {
            if (!num.empty()) { out.push_back((uint32_t)stoul(num)); num.clear(); }
        } else num.push_back(c);
    }
    if (!num.empty()) out.push_back((uint32_t)stoul(num));
    return !out.empty();
}

static bool readDocIDsFile(const string& path, vector<uint32_t>& out) {
    ifstream in(path);
    if (!in) return false;
    out.clear();
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        // allow CSV or one-per-line
        vector<uint32_t> tmp;
        if (parseCSVDocIDs(line, tmp)) {
            out.insert(out.end(), tmp.begin(), tmp.end());
        }
    }
    return !out.empty();
}

static bool parseArgs(int argc, char** argv, Args& a) {
    for (int i=1;i<argc;++i) {
        string s = argv[i];
        if (s=="--store-dir" && i+1<argc) a.storeDir = argv[++i];
        else if (s=="--query" && i+1<argc) a.query = argv[++i];
        else if (s=="--docs" && i+1<argc) { string d=argv[++i]; parseCSVDocIDs(d, a.docIDs); }
        else if (s=="--docs-file" && i+1<argc) a.docsFile = argv[++i];
        else if (s=="--windowsz" && i+1<argc) a.windowsz = stoul(argv[++i]);
        else if (s=="--per" && i+1<argc) a.perDoc = stoul(argv[++i]);
        else { usage(argv[0]); return false; }
    }
    if (a.query.empty()) { usage(argv[0]); return false; }
    if (a.docIDs.empty() && a.docsFile.empty()) { usage(argv[0]); return false; }
    if (!a.docsFile.empty() && !readDocIDsFile(a.docsFile, a.docIDs)) {
        cerr << "Failed to read docIDs from " << a.docsFile << "\n";
        return false;
    }
    return true;
}

// ---------------- PassageStore ----------------
struct PassageStore {
    vector<uint64_t> offsets; // size N+1
    string textsPath;
    ifstream fin;

    bool open(const string& dir) {
        string off = dir + "/offsets.bin";
        textsPath  = dir + "/texts.bin";
        ifstream in(off, ios::binary);
        if (!in) { cerr << "Cannot open " << off << "\n"; return false; }
        in.seekg(0, ios::end);
        size_t bytes = (size_t)in.tellg();
        if (bytes % sizeof(uint64_t) != 0) {
            cerr << "offsets.bin corrupt (size not multiple of 8)\n";
            return false;
        }
        size_t n = bytes / sizeof(uint64_t);
        offsets.resize(n);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(offsets.data()), bytes);
        fin.open(textsPath, ios::binary);
        if (!fin) { cerr << "Cannot open " << textsPath << "\n"; return false; }
        return true;
    }

    size_t size() const { return offsets.size() > 0 ? offsets.size()-1 : 0; }

    string get(uint32_t docID) {
        if (docID+1 >= offsets.size()) return {};
        uint64_t s = offsets[docID];
        uint64_t e = offsets[docID+1];
        if (e < s) return {};
        size_t len = size_t(e - s);
        string out; out.resize(len);
        fin.seekg((std::streamoff)s);
        fin.read(&out[0], len);
        return out;
    }
};

// ---------------- Text utils ----------------
struct Token {
    string norm;     // lowercased word
    size_t begin;    // char offset in original
    size_t end;      // char offset (exclusive)
};

static inline bool isWordChar(unsigned char c) {
    return std::isalnum(c) || c=='_' || c=='\''; // allow apostrophes
}

static vector<Token> tokenizeWithOffsets(const string& s) {
    vector<Token> tks;
    const size_t n = s.size();
    size_t i=0;
    while (i<n) {
        // skip non-word
        while (i<n && !isWordChar((unsigned char)s[i])) ++i;
        if (i>=n) break;
        size_t start=i;
        while (i<n && isWordChar((unsigned char)s[i])) ++i;
        size_t stop=i;
        string w = s.substr(start, stop-start);
        // normalize
        for (auto& ch: w) ch = (char)tolower((unsigned char)ch);
        tks.push_back({w, start, stop});
    }
    return tks;
}

static vector<string> splitQuery(const string& q) {
    vector<string> out;
    string cur;
    for (char c: q) {
        if (isspace((unsigned char)c)) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur.push_back((char)tolower((unsigned char)c));
        }
    }
    if (!cur.empty()) out.push_back(cur);
    // dedupe
    sort(out.begin(), out.end());
    out.erase(unique(out.begin(), out.end()), out.end());
    return out;
}

// ---------------- Scoring windows ----------------
// Build candidate windows centered at each match; score by:
//   score = 2*uniqueMatches + totalMatches + 0.5*(windowsz/spanTokens) + earlyBonus
// where earlyBonus = 0.2 if window begins within first 200 chars.
struct Window {
    size_t tokL=0, tokR=0; // inclusive token bounds
    size_t charL=0, charR=0; // char offsets in original
    int uniqueMatches=0;
    int totalMatches=0;
    double span = 1.0; // token span
    double score=0.0;
};

static vector<Window> pickWindows(const string& text,
                                  const vector<Token>& toks,
                                  const vector<string>& qterms,
                                  size_t windowsz,
                                  size_t want)
{
    unordered_set<string> qset(qterms.begin(), qterms.end());
    vector<size_t> matchPos;
    matchPos.reserve(toks.size());
    for (size_t i=0;i<toks.size();++i)
        if (qset.count(toks[i].norm)) matchPos.push_back(i);

    vector<Window> cand;
    cand.reserve(matchPos.size());
    for (size_t m: matchPos) {
        size_t half = windowsz/2;
        size_t L = (m>half? m-half: 0);
        size_t R = min(toks.size()? toks.size()-1:0, m+half);
        // compute coverage
        unordered_set<string> seen;
        int total=0;
        for (size_t i=L;i<=R && i<toks.size();++i) {
            if (qset.count(toks[i].norm)) { ++total; seen.insert(toks[i].norm); }
        }
        if (total==0) continue;

        Window w;
        w.tokL=L; w.tokR=R;
        w.charL = (L<toks.size()? toks[L].begin : 0);
        w.charR = (R<toks.size()? toks[R].end : text.size());
        w.uniqueMatches = (int)seen.size();
        w.totalMatches = total;
        w.span = double((R>=L)? (R-L+1): 1);
        double cov = 2.0 * w.uniqueMatches + 1.0 * w.totalMatches;
        double prox = 0.5 * (double(windowsz) / max(1.0, w.span));
        double early = (w.charL < 200 ? 0.2 : 0.0);
        w.score = cov + prox + early;
        cand.push_back(w);
    }

    // Sort by score desc, then earlier charL
    sort(cand.begin(), cand.end(), [](const Window& a, const Window& b){
        if (a.score!=b.score) return a.score>b.score;
        return a.charL < b.charL;
    });

    // take top 'want' non-overlapping windows
    vector<Window> out;
    for (auto& w: cand) {
        bool overlap=false;
        for (auto& v: out) {
            size_t L = max(w.charL, v.charL);
            size_t R = min(w.charR, v.charR);
            if (L < R) { overlap=true; break; }
        }
        if (!overlap) out.push_back(w);
        if (out.size()>=want) break;
    }
    return out;
}

// ---------------- Highlighting ----------------
// Simple word-boundary, case-insensitive highlighting: wrap with <b>..</b>.
// Works on the snippet slice (original casing preserved).
static string toLowerCopy(string s) {
    for (auto& c: s) c = (char)tolower((unsigned char)c);
    return s;
}

static string highlight(const string& snippet, const vector<string>& qterms) {
    if (snippet.empty()) return snippet;

    // Build lowercase copy for matching; then map matches to ranges
    string low = toLowerCopy(snippet);
    vector<pair<size_t,size_t>> ranges; // [start,end) of matches

    auto isWordBoundary = [&](int idx)->bool{
        if (idx<=0 || idx>=(int)snippet.size()) return true;
        unsigned char c = (unsigned char)snippet[idx-1];
        return !isalnum(c) && c!='_' && c!='\'';
    };
    auto isWordBoundaryR = [&](int idx)->bool{
        if (idx<0 || idx>=(int)snippet.size()-1) return true;
        unsigned char c = (unsigned char)snippet[idx+1];
        return !isalnum(c) && c!='_' && c!='\'';
    };

    for (auto& q : qterms) {
        if (q.empty()) continue;
        string ql = toLowerCopy(q);
        size_t pos = 0;
        while (true) {
            pos = low.find(ql, pos);
            if (pos == string::npos) break;
            // check word-ish boundaries
            if (isWordBoundary((int)pos) && isWordBoundaryR((int)(pos + ql.size() - 1))) {
                ranges.emplace_back(pos, pos + ql.size());
            }
            pos += ql.size();
        }
    }

    if (ranges.empty()) return snippet;

    // Merge overlapping ranges
    sort(ranges.begin(), ranges.end());
    vector<pair<size_t,size_t>> merged;
    for (auto& r: ranges) {
        if (merged.empty() || r.first > merged.back().second) merged.push_back(r);
        else merged.back().second = max(merged.back().second, r.second);
    }

    // Emit with <b>...</b>
    string out; out.reserve(snippet.size() + merged.size()*7);
    size_t cursor = 0;
    for (auto& r: merged) {
        if (r.first > cursor) out.append(snippet.substr(cursor, r.first - cursor));
        out.append("<b>");
        out.append(snippet.substr(r.first, r.second - r.first));
        out.append("</b>");
        cursor = r.second;
    }
    if (cursor < snippet.size()) out.append(snippet.substr(cursor));
    return out;
}

// ---------------- Main flow ----------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    // Open store
    PassageStore store;
    if (!store.open(args.storeDir)) return 2;

    // Prepare query terms
    vector<string> qterms = splitQuery(args.query);
    if (qterms.empty()) { cerr << "No valid query terms.\n"; return 3; }

    // Process each docID
    for (uint32_t doc : args.docIDs) {
        string text = store.get(doc);
        if (text.empty()) {
            // still emit empty
            cout << "{ \"docID\": " << doc << ", \"snippets\": [] }\n";
            continue;
        }

        // Tokenize & pick windows
        auto toks = tokenizeWithOffsets(text);
        auto wins = pickWindows(text, toks, qterms, args.windowsz, args.perDoc);

        vector<string> snippets;
        for (auto& w : wins) {
            // Extract a slice, add ellipses if we trimmed
            size_t L = w.charL, R = w.charR;
            // extend a little to next punctuation for nicer edges
            auto left = (L>10 ? L-10 : 0);
            auto right = min(text.size(), R+10);
            string slice = text.substr(left, right-left);

            // Tighten the slice to avoid starting/ending mid-word
            // (optional: already fairly readable)
            if (left>0) slice = string("… ") + slice;
            if (right < text.size()) slice += " …";

            // Highlight
            snippets.push_back( highlight(slice, qterms) );
        }

        // Fallback: if no windows matched (e.g., OCR noise), take head
        if (snippets.empty()) {
            string head = text.substr(0, min<size_t>(300, text.size()));
            if (head.size() < text.size()) head += " …";
            snippets.push_back(head);
        }

        // Emit JSON line
        cout << "{ \"docID\": " << doc << ", \"snippets\": [";
        for (size_t i=0;i<snippets.size();++i) {
            // naive JSON escaping for quotes/backslashes
            string s = snippets[i];
            string esc; esc.reserve(s.size()+16);
            for (char c: s) {
                if (c=='\\') esc += "\\\\";
                else if (c=='\"') esc += "\\\"";
                else if (c=='\n') esc += "\\n";
                else if (c=='\r') esc += "\\r";
                else esc += c;
            }
            if (i) cout << ", ";
            cout << "\"" << esc << "\"";
        }
        cout << "] }\n";
    }

    return 0;
}