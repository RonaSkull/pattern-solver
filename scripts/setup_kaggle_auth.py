"""
Kaggle Authentication Helper for ARC-AGI-3 Competition
"""
import os
import sys
import json
from pathlib import Path

def setup_kaggle_auth():
    """Setup Kaggle API authentication."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✅ Kaggle authentication already configured")
        return True
    
    print("=" * 60)
    print("KAGGLE API AUTHENTICATION REQUIRED")
    print("=" * 60)
    print("\n1. Go to: https://www.kaggle.com/settings")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New Token'")
    print("4. Download the kaggle.json file")
    print(f"5. Place it at: {kaggle_json}")
    print("\nOr enter credentials manually:")
    
    username = input("\nKaggle Username: ").strip()
    key = input("Kaggle API Key: ").strip()
    
    if username and key:
        auth_data = {"username": username, "key": key}
        with open(kaggle_json, 'w') as f:
            json.dump(auth_data, f)
        
        # Secure the file
        os.chmod(kaggle_json, 0o600)
        print(f"\n✅ Authentication saved to {kaggle_json}")
        return True
    else:
        print("\n❌ No credentials provided")
        return False

def download_competition_data():
    """Download ARC-AGI-3 competition data."""
    import subprocess
    
    comp_name = "arc-prize-2026-arc-agi-3"
    
    print(f"\n📥 Downloading {comp_name}...")
    
    result = subprocess.run(
        ["python", "-m", "kaggle", "competitions", "download", "-c", comp_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Download complete!")
        
        # Extract zip
        import zipfile
        zip_file = Path(f"{comp_name}.zip")
        
        if zip_file.exists():
            print(f"📂 Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall("data/")
            print("✅ Extracted to data/")
            return True
    else:
        print(f"❌ Error: {result.stderr}")
        return False

if __name__ == "__main__":
    if setup_kaggle_auth():
        download_competition_data()
    else:
        sys.exit(1)
