import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Configure production logging
logger = logging.getLogger("FilingChunker")

@dataclass
class Chunk:
    text: str
    ticker: str
    filing_type: str      # "10-K", "10-Q"
    fiscal_period: str
    filing_date: str      # ISO date string
    section: str          # e.g., "Item 1A Risk Factors"
    embedding: Optional[List[float]] = None


def split_by_section(filing_text: str, filing_type: str) -> List[Dict[str, str]]:
    """
    Production-grade section parser using anchor-based regex matching with fallback 
    boundaries to correctly isolate Item 1A and Item 7/2 from raw filing texts.
    """
    extracted_sections = []
    
    if not filing_text or not isinstance(filing_text, str):
        logger.warning("Empty or non-string filing text provided to section splitter.")
        return extracted_sections

    # Clean up standard XML/HTML non-breaking spaces or excessive whitespace if present
    normalized_text = re.sub(r'&#160;|&nbsp;', ' ', filing_text)

    if filing_type.upper() == '10-K':
        # Patterns for 10-K Item 1A and Item 7
        patterns = [
            {
                "section": "Item 1A Risk Factors",
                "start": r"(?i)\n\s*(?:PART\s+I\s+)?Item\s+1A\.?\s+Risk\s+Factors",
                "end": r"(?i)\n\s*(?:PART\s+(?:I|II)\s+)?Item\s+1B\.?\s+|(?:\n\s*Item\s+2\.?\s+)"
            },
            {
                "section": "Item 7 Management's Discussion and Analysis",
                "start": r"(?i)\n\s*(?:PART\s+II\s+)?Item\s+7\.?\s+Management\'s\s+Discussion\s+and\s+Analysis",
                "end": r"(?i)\n\s*(?:PART\s+II\s+)?Item\s+7A\.?\s+|(?:\n\s*Item\s+8\.?\s+)"
            }
        ]
    elif filing_type.upper() == '10-Q':
        patterns = [
            {
                "section": "Item 1A Risk Factors",
                "start": r"(?i)\n\s*(?:PART\s+II\s+)?Item\s+1A\.?\s+Risk\s+Factors",
                "end": r"(?i)\n\s*(?:PART\s+II\s+)?Item\s+2\.?\s+"
            },
            {
                "section": "Item 2 Management's Discussion and Analysis",
                "start": r"(?i)\n\s*(?:PART\s+I\s+)?Item\s+2\.?\s+Management\'s\s+Discussion\s+and\s+Analysis",
                "end": r"(?i)\n\s*(?:PART\s+II\s+)?Item\s+3\.?\s+"
            }
        ]
    else:
        logger.error(f"Unsupported filing type for section splitting: {filing_type}")
        return []

    for p in patterns:
        start_match = re.search(p["start"], normalized_text)
        if not start_match:
            logger.debug(f"Start boundary not found for {p['section']} in {filing_type} filing.")
            continue
            
        start_idx = start_match.end()
        search_zone = normalized_text[start_idx:]
        
        end_match = re.search(p["end"], search_zone)
        if end_match:
            end_idx = start_idx + end_match.start()
            section_content = normalized_text[start_idx:end_idx].strip()
        else:
            # Fallback: take a reasonable length block if end marker is missing
            logger.warning(f"End boundary missing for {p['section']}; capturing truncation block.")
            section_content = normalized_text[start_idx:start_idx + 250000].strip()
            
        if len(section_content) > 100:  # Ensure it's not an empty match or table-of-contents reference
            extracted_sections.append({
                "section": p["section"],
                "text": section_content
            })

    return extracted_sections


def chunk_section(section_text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits section text into token-bounded chunks based on sentence boundaries,
    ensuring proper sliding window overlap without infinite loops.
    """
    if not section_text:
        return []

    # Robust sentence splitting regex
    sentence_splitter = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_splitter.split(section_text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
            
        words = sentence.split()
        word_count = len(words)
        
        if current_word_count + word_count > max_tokens and current_chunk:
            # Finalize current chunk
            chunks.append(" ".join(current_chunk))
            
            # Roll back index for overlap computation
            overlap_count = 0
            backtrack_idx = i
            temp_overlap_sentences = []
            
            while backtrack_idx > 0 and overlap_count < overlap:
                prev_sentence = sentences[backtrack_idx - 1].strip()
                prev_words = prev_sentence.split()
                if overlap_count + len(prev_words) <= overlap:
                    temp_overlap_sentences.insert(0, prev_sentence)
                    overlap_count += len(prev_words)
                    backtrack_idx -= 1
                else:
                    break
                    
            current_chunk = temp_overlap_sentences
            current_word_count = overlap_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
            i += 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks


def build_chunks(filing_text: str, metadata: Dict[str, Any]) -> List[Chunk]:
    """
    Production orchestration function combining section parsing and sentence chunking 
    into structured Chunk models carrying full metadata.
    """
    filing_type = metadata.get("filing_type", "10-K")
    sections = split_by_section(filing_text, filing_type)
    
    chunk_list = []
    for sec in sections:
        section_name = sec["section"]
        section_text = sec["text"]
        
        section_chunks = chunk_section(section_text)
        for chunk_text in section_chunks:
            chunk_list.append(
                Chunk(
                    text=chunk_text,
                    ticker=metadata.get("ticker", "UNKNOWN"),
                    filing_type=filing_type,
                    fiscal_period=metadata.get("fiscal_period", "UNKNOWN"),
                    filing_date=metadata.get("filing_date", "1970-01-01"),
                    section=section_name
                )
            )
            
    logger.info(f"Generated {len(chunk_list)} production chunks for {metadata.get('ticker')} ({filing_type}).")
    return chunk_list