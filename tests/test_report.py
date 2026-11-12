import os
import re
import pytest

# Resolve the report path.
REPORT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'report.md'))

def get_report_content():
    if not os.path.exists(REPORT_PATH):
        raise AssertionError(f"report.md was not found at {REPORT_PATH}")
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        return f.read()

def get_section_discussion_answer(question_keyword, content):
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if question_keyword in line and '**' in line:
            answer_lines = []
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                # Stop if we hit a new list item, header, or divider
                if next_line.startswith('-') or next_line.startswith('##') or next_line.startswith('---'):
                    break
                if next_line:
                    answer_lines.append(next_line)
            return '\n'.join(answer_lines)
    return None

@pytest.mark.weight(5)
@pytest.mark.number("2.1")
def test_student_information():
    """report.md: Student Information check"""
    content = get_report_content()
    name_match = re.search(r'-\s*\*\*Name:\*\*\s*(.*)', content)
    mmdt_id_match = re.search(r'-\s*\*\*MMDT ID:\*\*\s*(.*)', content)
    
    assert name_match is not None, "Name field not found under Student Information"
    assert mmdt_id_match is not None, "MMDT ID field not found under Student Information"
    
    name = name_match.group(1).strip()
    mmdt_id = mmdt_id_match.group(1).strip()
    
    assert name and name != "[Write your Name here]", "Student Name must be filled out"
    assert not (name.startswith("[") and name.endswith("]")), "Student Name cannot contain placeholder text"
    
    assert mmdt_id and mmdt_id != "[Write your MMDT ID here]", "MMDT ID must be filled out"
    assert not (mmdt_id.startswith("[") and mmdt_id.endswith("]")), "MMDT ID cannot contain placeholder text"

@pytest.mark.weight(5)
@pytest.mark.number("2.2")
def test_dataset_configuration():
    """report.md: Dataset & Corpus Configuration check (minimum 50 claims, custom corpus)"""
    content = get_report_content()
    docs_match = re.search(r'-\s*\*\*Total Corpus Documents Configured:\*\*\s*(.*)', content)
    claims_match = re.search(r'-\s*\*\*Total Labeled Claims Configured:\*\*\s*(.*)', content)
    test_match = re.search(r'-\s*\*\*Total Test Claims Configured:\*\*\s*(.*)', content)
    total_match = re.search(r'-\s*\*\*Total Claims in Dataset:\*\*\s*(.*)', content)
    
    assert docs_match is not None, "Total Corpus Documents Configured field not found"
    assert claims_match is not None, "Total Labeled Claims Configured field not found"
    assert test_match is not None, "Total Test Claims Configured field not found"
    assert total_match is not None, "Total Claims in Dataset field not found"
    
    d_str = docs_match.group(1).strip()
    c_str = claims_match.group(1).strip()
    t_str = test_match.group(1).strip()
    tot_str = total_match.group(1).strip()
    
    assert not (d_str.startswith("[") and d_str.endswith("]")), "Total Corpus Documents Configured cannot contain placeholder text"
    assert not (c_str.startswith("[") and c_str.endswith("]")), "Total Labeled Claims Configured cannot contain placeholder text"
    assert not (t_str.startswith("[") and t_str.endswith("]")), "Total Test Claims Configured cannot contain placeholder text"
    assert not (tot_str.startswith("[") and tot_str.endswith("]")), "Total Claims in Dataset cannot contain placeholder text"
    
    d_match = re.search(r'\d+', d_str)
    c_match = re.search(r'\d+', c_str)
    t_match = re.search(r'\d+', t_str)
    tot_match = re.search(r'\d+', tot_str)
    
    assert d_match is not None, "Total Corpus Documents Configured must contain a numeric value"
    assert c_match is not None, "Total Labeled Claims Configured must contain a numeric value"
    assert t_match is not None, "Total Test Claims Configured must contain a numeric value"
    assert tot_match is not None, "Total Claims in Dataset must contain a numeric value"
    
    d_val = int(d_match.group(0))
    c_val = int(c_match.group(0))
    t_val = int(t_match.group(0))
    tot_val = int(tot_match.group(0))
    
    assert d_val >= 20, f"You must build a custom Myanmar corpus of at least 20 reference documents (currently: {d_val})"
    assert c_val >= 40, f"You must configure at least 40 labeled claims in your Myanmar dataset (currently: {c_val})"
    assert t_val >= 10, f"You must configure at least 10 test claims in your Myanmar dataset (currently: {t_val})"
    assert tot_val >= 50, f"You must curate a dataset with at least 50 total Myanmar claims (currently: {tot_val})"
    assert (c_val + t_val) >= 50, f"The sum of labeled and test claims must be at least 50 (currently: {c_val + t_val})"

@pytest.mark.weight(5)
@pytest.mark.number("2.3")
def test_local_verification_checked():
    """report.md: Verification checklist"""
    content = get_report_content()
    checked_algs = re.findall(r'-\s*\[[xX]\]\s*(.*)', content)
    assert len(checked_algs) > 0, "No algorithms or features were checked off as verified (marked with [x])"

@pytest.mark.weight(5)
@pytest.mark.number("2.4")
def test_presentation_information():
    """report.md: Presentation & Project Structure check"""
    content = get_report_content()
    video_match = re.search(r'-\s*\*\*Video Presentation Link:\*\*\s*(.*)', content)
    struct_match = re.search(r'-\s*\*\*Project Directory Structure:\*\*\s*(.*)', content)
    
    assert video_match is not None, "Video Presentation Link field not found"
    assert struct_match is not None, "Project Directory Structure field not found"
    
    video_link = video_match.group(1).strip()
    struct_desc = struct_match.group(1).strip()
    
    assert video_link and not video_link.startswith("["), "Video Presentation Link must be completed and cannot be placeholder"
    assert video_link.startswith("http://") or video_link.startswith("https://"), "Video Presentation Link must be a valid absolute HTTP/HTTPS URL"
    assert struct_desc and not struct_desc.startswith("["), "Project Directory Structure must be completed and cannot be placeholder"

@pytest.mark.weight(10)
@pytest.mark.number("2.5")
def test_formulation_discussion():
    """report.md: Myanmar NLP Formulation and Discussion check"""
    content = get_report_content()
    discussion_points = [
        "How did you formulate Myanmar text preprocessing (syllable/word segmentation, Unicode normalization, stopword filtering), TF-IDF feature extraction, and the Naïve Bayes probabilistic classifier?",
        "Compare Traditional NLP (TF-IDF + Naïve Bayes) vs Dense Semantic Retrieval & RAG on Myanmar text in terms of vocabulary coverage, morphology handling, and stance classification (SUPPORTS, REFUTES, NOT ENOUGH INFO).",
        "How does RAG help mitigate LLM hallucinations during claim verification on low-resource/Myanmar text, and how does evidence retrieval quality affect the grounded explanations?"
    ]
    
    for idx, question in enumerate(discussion_points, 1):
        text = get_section_discussion_answer(question, content)
        assert text is not None, f"Discussion question {idx} ('{question}') is missing from report.md"
        text = text.strip()
        assert text != "", f"Discussion question {idx} ('{question}') has not been filled out (it is empty)"
        assert not (text.startswith("[") and text.endswith("]")), f"Discussion question {idx} ('{question}') cannot contain placeholder text"
