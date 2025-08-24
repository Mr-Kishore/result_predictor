def test_folder_structure():
    import os
    required_folders = [
        'app', 'app/controllers', 'app/managers', 'app/models', 'app/routes', 'app/services', 'app/utils', 'app/chatbot',
        'data', 'model', 'static', 'templates', 'uploads', 'tests'
    ]
    for folder in required_folders:
        assert os.path.isdir(folder), f"Missing folder: {folder}"

def test_env_file():
    import os
    assert os.path.isfile('.env'), ".env file missing"

def test_gitignore():
    with open('.gitignore') as f:
        content = f.read()
    assert '__pycache__/' in content, "__pycache__/ not in .gitignore"
    assert '.env' in content, ".env not in .gitignore"
