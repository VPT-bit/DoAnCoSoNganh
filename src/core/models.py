"""Core data models for the chatbot."""
from dataclasses import dataclass, field
from typing import Optional
import json
import uuid


@dataclass
class Chunk:
    """Represents a text chunk from a document."""
    file_id: str
    file_name: str
    chunk_id: int
    refs: list[str]
    text: str
    
    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "chunk_id": self.chunk_id,
            "refs": self.refs,
            "text": self.text,
        }
    
    def format_for_llm(self) -> str:
        """Format chunk with full metadata for LLM context."""
        refs_str = ", ".join(self.refs[:5])  # Limit refs for readability
        if len(self.refs) > 5:
            refs_str += f" (+{len(self.refs) - 5} more)"
        return (
            f"[Chunk {self.chunk_id} | File: {self.file_name} | ID: {self.file_id}]\n"
            f"[Refs: {refs_str}]\n"
            f"{self.text}"
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            file_id=data["file_id"],
            file_name=data["file_name"],
            chunk_id=data["chunk_id"],
            refs=data.get("refs", []),
            text=data["text"],
        )


@dataclass
class FileSummary:
    """Summary information for a processed file."""
    file_id: str
    file_name: str
    summary: str
    chunk_count: int
    
    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "summary": self.summary,
            "chunk_count": self.chunk_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FileSummary":
        return cls(
            file_id=data["file_id"],
            file_name=data["file_name"],
            summary=data.get("summary", ""),
            chunk_count=data.get("chunk_count", 0),
        )


@dataclass
class Corpus:
    """
    Container for all document data.
    Holds chunks and file summaries for RAG operations.
    """
    chunks: list[Chunk] = field(default_factory=list)
    summaries: list[FileSummary] = field(default_factory=list)
    
    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the corpus."""
        self.chunks.append(chunk)
    
    def add_summary(self, summary: FileSummary) -> None:
        """Add a file summary to the corpus."""
        self.summaries.append(summary)
    
    def get_summary_by_file_id(self, file_id: str) -> Optional[FileSummary]:
        """Get summary for a specific file."""
        for summary in self.summaries:
            if summary.file_id == file_id:
                return summary
        return None
    
    def get_chunks_by_file_id(self, file_id: str) -> list[Chunk]:
        """Get all chunks for a specific file."""
        return [c for c in self.chunks if c.file_id == file_id]
    
    def get_chunks_by_file_ids(self, file_ids: list[str]) -> list[Chunk]:
        """Get all chunks for multiple files."""
        file_id_set = set(file_ids)
        return [c for c in self.chunks if c.file_id in file_id_set]
    
    def get_file_ids(self) -> list[str]:
        """Get list of all file IDs."""
        return [s.file_id for s in self.summaries]
    
    def get_file_by_name_fuzzy(self, name_hint: str) -> FileSummary | None:
        """Find file by fuzzy name match."""
        name_lower = name_hint.lower()
        for s in self.summaries:
            if name_lower in s.file_name.lower():
                return s
        return None
    
    def remove_file(self, file_id: str) -> None:
        """Remove all data for a specific file."""
        self.chunks = [c for c in self.chunks if c.file_id != file_id]
        self.summaries = [s for s in self.summaries if s.file_id != file_id]
    
    def extract_texts(self) -> list[str]:
        """Extract all chunk texts."""
        return [chunk.text for chunk in self.chunks]
    
    def extract_texts_by_indices(self, indices: list[int]) -> list[str]:
        """Extract chunk texts by indices."""
        return [self.chunks[i].text for i in indices if i < len(self.chunks)]
    
    def extract_formatted_chunks(self) -> list[str]:
        """Extract all chunks formatted with metadata for LLM."""
        return [chunk.format_for_llm() for chunk in self.chunks]
    
    def extract_formatted_by_indices(self, indices: list[int]) -> list[str]:
        """Extract formatted chunks by indices."""
        return [self.chunks[i].format_for_llm() for i in indices if i < len(self.chunks)]
    
    def extract_by_indices(self, indices: list[int]) -> "Corpus":
        """Create a new corpus with chunks at given indices."""
        new_corpus = Corpus()
        for i in indices:
            if i < len(self.chunks):
                new_corpus.add_chunk(self.chunks[i])
        return new_corpus
    
    def get_all_summaries_text(self) -> str:
        """Get formatted text of all file summaries."""
        if not self.summaries:
            return ""
        parts = []
        for s in self.summaries:
            parts.append(f"[{s.file_name}]\n{s.summary}")
        return "\n\n".join(parts)
    
    def get_file_list(self) -> list[dict]:
        """Get list of files with their info."""
        return [
            {"file_id": s.file_id, "file_name": s.file_name, "chunk_count": s.chunk_count}
            for s in self.summaries
        ]
    
    def is_empty(self) -> bool:
        """Check if corpus has no data."""
        return len(self.chunks) == 0
    
    def to_dict(self) -> dict:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "summaries": [s.to_dict() for s in self.summaries],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Corpus":
        corpus = cls()
        for c in data.get("chunks", []):
            corpus.add_chunk(Chunk.from_dict(c))
        for s in data.get("summaries", []):
            corpus.add_summary(FileSummary.from_dict(s))
        return corpus
    
    def save(self, filepath: str) -> None:
        """Save corpus to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "Corpus":
        """Load corpus from JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()


def generate_file_id() -> str:
    """Generate unique file ID."""
    return str(uuid.uuid4())[:8]
