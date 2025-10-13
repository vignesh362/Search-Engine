#!/usr/bin/env python3
"""
Create minimal passage store for testing when collection.tsv is missing
"""

import os
import struct

def create_minimal_passage_store():
    """Create minimal passage store files for testing"""
    
    # Create some dummy documents for testing
    test_docs = [
        "The quick brown fox jumps over the lazy dog. This is a test document about animals.",
        "Computer science is the study of computational systems and programming languages.",
        "Machine learning and artificial intelligence are transforming technology.",
        "The university offers programs in engineering, medicine, and business administration.",
        "Cats and dogs are popular pets around the world. Many families have both.",
        "Research shows that exercise and healthy eating improve quality of life.",
        "The internet has revolutionized communication and information sharing.",
        "Climate change is one of the most pressing challenges of our time.",
        "Education plays a crucial role in personal development and career success.",
        "Technology continues to evolve rapidly, changing how we work and live."
    ]
    
    offsets_path = "index_output/offsets.bin"
    texts_path = "index_output/texts.bin"
    
    print("Creating minimal passage store for testing...")
    
    # Create texts.bin
    offsets = []
    current_offset = 0
    
    with open(texts_path, 'wb') as texts_file:
        for i, doc in enumerate(test_docs):
            offsets.append(current_offset)
            doc_bytes = doc.encode('utf-8')
            texts_file.write(doc_bytes)
            current_offset += len(doc_bytes)
            print(f"  Added document {i}: {len(doc_bytes)} bytes")
    
    # Add sentinel offset
    offsets.append(current_offset)
    
    # Create offsets.bin
    with open(offsets_path, 'wb') as offsets_file:
        for offset in offsets:
            offsets_file.write(struct.pack('<Q', offset))  # Little-endian uint64
    
    print(f"Created {len(test_docs)} test documents")
    print(f"Total text size: {current_offset} bytes")
    print(f"Offsets file: {len(offsets)} entries")
    print("\nPassage store ready for testing!")

if __name__ == "__main__":
    create_minimal_passage_store()
