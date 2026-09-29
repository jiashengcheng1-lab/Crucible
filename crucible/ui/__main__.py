"""`python -m crucible.ui` or the `crucible-ui` script: launch the Streamlit app."""
import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        raise SystemExit("Streamlit is not installed: pip install 'crucible[ui]'")
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
