import os
import sys
import glob
import pytest

@pytest.mark.weight(10)
@pytest.mark.number("1.1")
def test_submitted_files(request):
    """Check that required core files and project structure exist"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    expected_files = [
        'app.py',
        'report.md',
        'AI_declaration.png',
        'data'
    ]
    
    missing_files = []
    existing_files = []
    for relative_path in expected_files:
        full_path = os.path.join(project_root, relative_path)
        if not os.path.exists(full_path):
            missing_files.append(relative_path)
        else:
            existing_files.append(relative_path)
            
    earned = int(round(10.0 * len(existing_files) / len(expected_files)))
    request.node._earned_points = earned
    
    assert len(missing_files) == 0, f"Missing required files under project directory: {missing_files}"
    print("All core files and project structure present!")

@pytest.mark.weight(30)
@pytest.mark.number("1.2")
def test_verifier_generates_predictions(request):
    """Test that studentverifier() in app.py executes and generates <studentid>_predictions.csv"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    try:
        import app
    except ImportError as e:
        pytest.fail(f"Could not import app.py: {e}")
        
    if not hasattr(app, 'studentverifier') or not callable(getattr(app, 'studentverifier')):
        pytest.fail("app.py must define a callable function named 'studentverifier()'")
        
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'data')
    
    # Ensure sample test data exists if running in a fresh environment without local datasets
    sample_corpus_path = os.path.join(data_dir, 'corpus.csv')
    sample_train_path = os.path.join(data_dir, 'train_claims.csv')
    sample_test_path = os.path.join(data_dir, 'test_claims.csv')
    
    cleanup_temp_data = False
    if not os.path.exists(sample_corpus_path) or not os.path.exists(sample_test_path):
        cleanup_temp_data = True
        os.makedirs(data_dir, exist_ok=True)
        with open(sample_corpus_path, 'w', encoding='utf-8') as f:
            f.write("doc_id,title,text,source\n")
            f.write("DOC_001,ပုဂံ,ပုဂံသည် မြန်မာနိုင်ငံတွင် တည်ရှိသော ရှေးဟောင်းယဉ်ကျေးမှုနယ်မြေဖြစ်သည်။,သမိုင်း\n")
            f.write("DOC_002,ရွှေတိဂုံ,ရွှေတိဂုံစေတီတော်သည် ရန်ကုန်မြို့တွင် တည်ရှိသည်။,သာသနာ\n")
        with open(sample_train_path, 'w', encoding='utf-8') as f:
            f.write("claim_id,claim_text,evidence_id,label\n")
            f.write("CLM_001,ပုဂံသည် မြန်မာနိုင်ငံတွင် တည်ရှိသည်။,DOC_001,SUPPORTS\n")
            f.write("CLM_002,ပုဂံသည် ဂျပန်တွင် တည်ရှိသည်။,DOC_001,REFUTES\n")
        with open(sample_test_path, 'w', encoding='utf-8') as f:
            f.write("claim_id,claim_text\n")
            f.write("CLM_TEST_001,ပုဂံသည် မြန်မာနိုင်ငံတွင် တည်ရှိသည်။\n")
            f.write("CLM_TEST_002,ရွှေတိဂုံသည် ရန်ကုန်တွင် တည်ရှိသည်။\n")
    
    # Run the student verifier
    try:
        result = app.studentverifier(data_dir=data_dir, output_dir=output_dir)
    except Exception as e:
        pytest.fail(f"studentverifier() failed during execution with error: {e}")
    finally:
        if cleanup_temp_data:
            for p in [sample_corpus_path, sample_train_path, sample_test_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        
    # Check if predictions CSV exists
    # First check returned path if valid string
    csv_file = None
    if isinstance(result, str) and os.path.exists(result) and result.endswith('.csv'):
        csv_file = result
    else:
        # Check in output_dir for any *_predictions.csv
        matching = glob.glob(os.path.join(output_dir, "*_predictions.csv"))
        if not matching:
            # Check project root as fallback
            matching = glob.glob(os.path.join(project_root, "*_predictions.csv"))
        if matching:
            csv_file = matching[0]
            
    assert csv_file is not None and os.path.exists(csv_file), (
        "studentverifier() did not generate any '<studentid>_predictions.csv' file in data/ folder. "
        f"Return value was: {result}"
    )
    
    # Check that the file is not empty
    file_size = os.path.getsize(csv_file)
    assert file_size > 0, f"Generated predictions CSV '{os.path.basename(csv_file)}' is empty (0 bytes)"
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    assert len(lines) >= 2, (
        f"Generated predictions CSV '{os.path.basename(csv_file)}' should contain a header and prediction rows (found {len(lines)} lines)"
    )
    
    print(f"Successfully verified generated predictions CSV: {csv_file} ({len(lines)} lines)")
