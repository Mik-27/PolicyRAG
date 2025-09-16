import re
from typing import List, Dict
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunker:
    """
    Recursive text chunker using LangChain's RecursiveCharacterTextSplitter.
    
    Splitting strategy (in order):
    1. Paragraphs (double newlines)
    2. Line breaks
    3. Sentences (with period + space)
    4. Words (with space)
    5. Character fallback
    
    This respects document hierarchy while ensuring chunks stay under token limit.
    """
    
    def __init__(self, 
                 max_tokens: int = 512, 
                 min_tokens: int = 100, 
                 overlap_tokens: int = 50,
                 model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize the RecursiveChunker.
        
        Args:
            max_tokens: Maximum tokens per chunk (should match model context size)
            min_tokens: Minimum tokens to consider a valid chunk
            overlap_tokens: Number of tokens to overlap between chunks for context
            model_name: Tokenizer model to use for token counting
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Initialize LangChain's recursive splitter
        # Separators ordered from most to least significant
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            length_function=self._count_tokens,
            separators=[
                "\n\n",        # Paragraph breaks
                "\n",          # Line breaks
                ". ",          # Sentences
                " ",           # Words
                ""             # Character fallback
            ],
            add_start_index=False,
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def _extract_section_header(self, text: str) -> str:
        """
        Extract section header from text.
        Returns the first line if it looks like a header, otherwise returns "General".
        """
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if first line looks like a header (markdown or numbered section)
            if re.match(r'^(#{1,6}\s+|^\d+(?:\.\d+)*\s+)', first_line):
                return first_line[:100]  # Limit to 100 chars
        return "General"
    
    def chunk(self, text: str, page_number: int = 1) -> List[Dict[str, any]]:
        """
        Main chunking method. Recursively chunks text and returns structured chunks.
        
        Args:
            text: Text to chunk (can be a full page or entire document)
            page_number: Page number for metadata tracking
        
        Returns:
            List of chunk dictionaries with structure:
            {
                'content': str,           # The actual chunk text
                'page': int,              # Page number
                'section': str,           # Section header (if found)
                'chunk_index': int,       # Index within this page
                'token_count': int,       # Token count of this chunk
            }
        """
        # Clean up text
        text = text.strip()
        if not text:
            return []
        
        # Perform recursive splitting using LangChain
        split_chunks = self.splitter.split_text(text)
        
        # Create structured chunks with metadata
        result = []
        for idx, chunk in enumerate(split_chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            
            token_count = self._count_tokens(chunk)
            
            # Skip chunks that are too small (likely noise)
            if token_count < self.min_tokens // 2:
                continue
            
            chunk_dict = {
                'content': chunk,
                'page': page_number,
                'section': self._extract_section_header(chunk),
                'chunk_index': idx,
                'token_count': token_count,
            }
            result.append(chunk_dict)
        
        return result
    
    def chunk_document(self, text_list: List[str]) -> List[Dict[str, any]]:
        """
        Chunk multiple pages of text (typically from pdf_to_text()).
        
        Args:
            text_list: List of text strings (one per page)
        
        Returns:
            List of chunk dictionaries with page metadata
        """
        all_chunks = []
        
        for page_idx, page_text in enumerate(text_list, start=1):
            if not page_text or not page_text.strip():
                continue
            
            page_chunks = self.chunk(page_text, page_number=page_idx)
            all_chunks.extend(page_chunks)
        
        return all_chunks


def test_chunker():
    """Quick test of the chunker (run with: python chunking.py)"""
    sample_text = """
# Company Policy Manual

## Section 1: Leave Policy

### 1.1 Vacation Days
Employees are entitled to vacation days based on their tenure. First year employees receive 10 days. 
After 5 years, employees receive 15 days. After 10 years, employees receive 20 days.

### 1.2 Sick Leave
Employees may take up to 10 sick days per year. A doctor's note is required for absences 
exceeding 3 consecutive days. Unused sick days do not roll over to the next year.

## Section 2: Remote Work Policy

Employees are permitted to work remotely up to 2 days per week with manager approval.
All remote work must maintain company security standards and VPN usage.
    """
    
    chunker = RecursiveChunker(max_tokens=100, min_tokens=20)
    chunks = chunker.chunk(sample_text)
    
    print(f"Generated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Section: {chunk['section']}")
        print(f"  Tokens: {chunk['token_count']}")
        print(f"  Content: {chunk['content'][:80]}...")
        print()


if __name__ == "__main__":
    test_chunker()
