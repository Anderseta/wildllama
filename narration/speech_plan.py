"""Plan pronunciation, paragraph direction and captions on the same timeline."""
import re

ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "e.g.", "i.e.", "vs.", "u.s."}


def paragraph_units(text, pronunciations):
    # Match whole phrases, longest first. Whitespace around punctuation stays visible.
    text = " ".join(text.split())
    choices = {" ".join(p["term"].split()).casefold(): p["spoken"].strip() for p in pronunciations}
    # Use the original spellings in the regexp (casefold may expand characters).
    terms = {" ".join(p["term"].split()) for p in pronunciations}
    pattern = "|".join(re.escape(k) for k in sorted(terms, key=len, reverse=True) if k)
    def tokens(fragment):
        return [[w, w] for w in fragment.split()]
    if not pattern:
        return tokens(text)
    units, last = [], 0
    for match in re.finditer(r"(?<!\w)(?:" + pattern + r")(?!\w)", text, re.I):
        units.extend(tokens(text[last:match.start()]))
        end = match.end()
        while end < len(text) and text[end] in ".,;:!?…":
            end += 1
        suffix = text[match.end():end]
        spoken = choices.get(match.group().casefold(), match.group())
        units.append([match.group() + suffix, spoken + suffix])
        last = end
    units.extend(tokens(text[last:]))
    return units


def speech_plan(cfg, limit=340):
    chunks = []
    paragraphs = [p.strip() for p in re.split(r"\r?\n+", cfg["text"]) if p.strip()]
    for index, paragraph in enumerate(paragraphs):
        units = paragraph_units(paragraph, cfg.get("pronunciations", []))
        while units:
            size, end, sentence, clause = 0, 0, 0, 0
            for display, spoken in units:
                if end and size + len(spoken) + 1 > limit:
                    break
                if len(spoken) > limit * 2:
                    raise ValueError("Separe as palavras longas do roteiro.")
                size += len(spoken) + 1
                end += 1
                token = display.rstrip("\"'”’)")
                if token.endswith((".", "!", "?", "…")) and token.lower() not in ABBREVIATIONS and not re.fullmatch(r"(?:[A-Za-z]\.)+", token):
                    sentence = end
                elif token.endswith((",", ";", ":","—")):
                    clause = end
            cut = end if end == len(units) else sentence or clause or end
            part, units = units[:cut], units[cut:]
            chunks.append({"paragraph": index, "units": part, "spoken": " ".join(w[1] for w in part),
                           "paragraph_end": not units and index + 1 < len(paragraphs)})
    if not chunks:
        raise ValueError("Digite o texto para narrar.")
    return chunks


def caption_times(units, marks, duration, offset):
    """Associate spoken spans with their original labels, including one-to-many aliases."""
    def weight(s):
        return sum(ch.isalnum() for ch in s)
    usable = [m for m in marks if weight(m.text)]
    if not usable:
        raise ValueError("O alinhador não encontrou palavras no trecho.")
    total = sum(weight(m.text) for m in usable)
    expected = sum(weight(spoken) for _, spoken in units)
    if not expected:
        return []
    cursor = 0
    edges = []
    for mark in usable:
        cursor += weight(mark.text)
        edges.append(cursor)
    from bisect import bisect_right, bisect_left
    result, consumed, prefix = [], 0, ""
    for label, spoken in units:
        n = weight(spoken)
        if not n:
            if result:
                result[-1][2] += " " + label
            else:
                prefix += label + " "
            continue
        a = min(len(usable)-1, bisect_right(edges, consumed * total / expected))
        consumed += n
        b = max(a, min(len(usable)-1, bisect_left(edges, consumed * total / expected)))
        start = max(0., min(duration, usable[a].start_time))
        end = max(start, min(duration, usable[b].end_time))
        result.append([offset+start, offset+end, prefix+label])
        prefix = ""
    return result
