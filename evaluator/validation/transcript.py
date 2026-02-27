"""
Transcript structure validation.
Raises ValueError if the transcript is missing required keys or has invalid turns.
"""


def validate_transcript(transcript: dict) -> None:
    """
    Validate transcript has required keys and valid turns.

    Required: transcript_id, turns (non-empty list).
    Each turn must have role and content.
    At least one turn must have role='agent'.

    Raises:
        ValueError: If transcript is invalid.
    """
    if not isinstance(transcript, dict):
        raise ValueError("transcript must be a dict")
    if "transcript_id" not in transcript:
        raise ValueError("transcript missing required key 'transcript_id'")
    if "turns" not in transcript:
        raise ValueError("transcript missing required key 'turns'")
    turns = transcript["turns"]
    if not isinstance(turns, list):
        raise ValueError("transcript 'turns' must be a list")
    if len(turns) == 0:
        raise ValueError("transcript 'turns' must not be empty")
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"turn at index {i} must be a dict, got {type(turn).__name__}")
        if "role" not in turn:
            raise ValueError(f"turn at index {i} missing required key 'role'")
        if "content" not in turn:
            raise ValueError(f"turn at index {i} missing required key 'content'")
    if not any(t.get("role") == "agent" for t in turns):
        raise ValueError("transcript must contain at least one turn with role='agent'")
