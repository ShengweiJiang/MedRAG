import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "src/app.py"]
    sys.exit(stcli.main())
