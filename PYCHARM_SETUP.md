# PyCharm Setup Guide

Complete guide for configuring PyCharm to use the virtual environment for this project.

## Quick Start (TL;DR)

1. Open PyCharm → **Open** → Select `mlcon-berlin-2025` folder
2. PyCharm will auto-detect `.venv` → Click **OK** to use it
3. Run any Python file with right-click → **Run**

---

## Detailed Setup

### Step 1: Open the Project

1. Launch **PyCharm**
2. Click **Open** (or **File → Open**)
3. Navigate to the `mlcon-berlin-2025` directory
4. Click **OK**

PyCharm will scan the project and typically auto-detect the existing `.venv` directory.

### Step 2: Configure Python Interpreter (Automatic)

**If PyCharm Detects .venv:**

1. You'll see a notification: **"Python interpreter is not configured"** or **"Virtualenv found"**
2. Click **Configure Python Interpreter** or **Use this virtualenv**
3. PyCharm will automatically set up `.venv` as the interpreter
4. Wait for PyCharm to index the project (progress bar at bottom)

**Verify It Worked:**
- Look at the bottom-right corner → Should show **Python 3.9 (.venv)**
- Open **File → Settings → Project → Python Interpreter** → Should show `.venv/bin/python`

### Step 3: Configure Python Interpreter (Manual)

**If PyCharm Doesn't Auto-Detect:**

1. **File → Settings** (Windows/Linux) or **PyCharm → Preferences** (macOS)
   - Or press `Cmd+,` (macOS) / `Ctrl+Alt+S` (Windows/Linux)

2. Navigate to **Project: mlcon-berlin-2025 → Python Interpreter**

3. Click the **gear icon** ⚙️ next to the interpreter dropdown

4. Select **Add Interpreter → Add Local Interpreter**

5. In the dialog:
   - Select **Virtualenv Environment** in the left panel
   - Choose **Existing environment**
   - Click the **folder icon** 📁 next to **Interpreter**
   - Navigate to: `<project-root>/.venv/bin/python` (macOS/Linux) or `<project-root>/.venv/Scripts/python.exe` (Windows)
   - Click **OK**

6. Wait for PyCharm to scan and index the packages (progress bar at bottom)

7. Click **OK** to close Settings

### Step 4: Verify Configuration

1. **Check Interpreter Badge:**
   - Look at the **bottom-right corner** of PyCharm
   - Should display: **Python 3.9 (.venv)**
   - If it shows system Python or wrong version, click it and select **.venv** from the list

2. **Check Installed Packages:**
   - **File → Settings → Project → Python Interpreter**
   - You should see all packages from `requirements.txt` listed (tensorflow, requests, ollama, etc.)
   - If the list is empty, proceed to Step 5

3. **Check Terminal:**
   - Open **Terminal** tool window (bottom toolbar or `Alt+F12`)
   - Should show `(.venv)` prefix in the prompt
   - Example: `(.venv) user@machine mlcon-berlin-2025 %`

### Step 5: Install Dependencies (If Needed)

**Method A: Using PyCharm UI (Easiest)**

1. Open `requirements.txt`
2. PyCharm will show a banner: **"Package requirements are not satisfied"**
3. Click **Install requirements**
4. Wait for installation to complete

**Method B: Using Terminal**

1. Open **Terminal** in PyCharm (`Alt+F12`)
2. Verify `(.venv)` prefix is shown
3. Run:
   ```bash
   pip install -r requirements.txt
   ```
4. Wait for installation to complete

**Method C: Using Package Manager**

1. **File → Settings → Project → Python Interpreter**
2. Click the **+** button (top-left of package list)
3. Search for packages individually and install
4. Or use **Install requirements from file** → Select `requirements.txt`

### Step 6: Run Python Scripts

**Method A: Right-Click Run (Easiest)**

1. Open any Python file (e.g., `day-1/ai_astrology_groq.py`)
2. Right-click anywhere in the editor
3. Select **Run 'ai_astrology_groq'**
4. Output appears in the **Run** tool window at the bottom

**Method B: Using the Run Button**

1. Open any Python file
2. Click the green ▶️ play button in the gutter next to `if __name__ == "__main__":`
3. Or press `Ctrl+Shift+R` (macOS) / `Shift+F10` (Windows/Linux)

**Method C: Create Run Configuration**

1. **Run → Edit Configurations**
2. Click **+** → **Python**
3. Configure:
   - **Name**: Your script name (e.g., "AI Astrology")
   - **Script path**: Browse to the Python file
   - **Python interpreter**: Should automatically show `.venv` interpreter
   - **Working directory**: Set to project root (usually auto-set)
   - **Environment variables**: Add if needed (e.g., API keys)
4. Click **OK**
5. Select your configuration from the dropdown (top-right) and click ▶️

**Method D: Run from Project View**

1. In the **Project** tool window (left sidebar)
2. Right-click any `.py` file
3. Select **Run 'filename'**

### Step 7: Debug Python Scripts

1. Set breakpoints by clicking left of line numbers (red dot appears)
2. Right-click the file → **Debug 'filename'**
   - Or press `Ctrl+Shift+D` (macOS) / `Shift+F9` (Windows/Linux)
3. Use the **Debug** tool window to:
   - Step through code (`F8` = step over, `F7` = step into)
   - Inspect variables
   - Evaluate expressions
   - View call stack

---

## Troubleshooting

### Issue: "No Python interpreter configured"

**Solution:**
- Follow **Step 3** above to manually configure the interpreter
- Make sure `.venv` directory exists in project root
- If `.venv` doesn't exist, create it:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate  # macOS/Linux
  # or
  .venv\Scripts\activate  # Windows
  pip install -r requirements.txt
  ```

### Issue: "ModuleNotFoundError" when running scripts

**Cause:** Dependencies not installed or wrong interpreter selected

**Solutions:**

1. **Check interpreter** (bottom-right) → Should show **Python 3.9 (.venv)**
   - If wrong, click it and select **.venv** from the list

2. **Install dependencies:**
   - Open `requirements.txt`
   - Click **Install requirements** banner
   - Or run in terminal: `pip install -r requirements.txt`

3. **Invalidate caches:**
   - **File → Invalidate Caches → Invalidate and Restart**
   - This forces PyCharm to re-scan packages

### Issue: urllib3 OpenSSL Warning

**Warning Message:**
```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, 
currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'
```

**Cause:** Using system Python instead of `.venv` interpreter

**Solution:**

1. **Verify interpreter:**
   - Check bottom-right corner → Should show **Python 3.9 (.venv)**
   - If it shows system Python, click it and select **.venv**

2. **Check requirements.txt:**
   - The `.venv` has `urllib3<2` which avoids this warning
   - If you see the warning, you're using the wrong interpreter

3. **Reinstall in venv:**
   ```bash
   source .venv/bin/activate  # Activate venv
   pip install --force-reinstall 'urllib3<2'
   ```

### Issue: Terminal doesn't show (.venv) prefix

**Solution:**

1. **File → Settings → Tools → Terminal**
2. Check **"Activate virtualenv"**
3. Click **OK**
4. Close and reopen the terminal window
5. Should now show `(.venv)` prefix

**Alternative:**
- Manually activate in terminal:
  ```bash
  source .venv/bin/activate  # macOS/Linux
  .venv\Scripts\activate     # Windows
  ```

### Issue: PyCharm is slow or laggy

**Solutions:**

1. **Increase memory:**
   - **Help → Edit Custom VM Options**
   - Increase `-Xmx` value (e.g., `-Xmx2048m`)
   - Restart PyCharm

2. **Exclude directories:**
   - Right-click `.venv` folder → **Mark Directory as → Excluded**
   - This prevents PyCharm from indexing venv internals
   - Also exclude: `__pycache__`, `.pytest_cache`, etc.

3. **Disable unnecessary plugins:**
   - **File → Settings → Plugins**
   - Disable plugins you don't use
   - Restart PyCharm

### Issue: Code completion not working

**Solutions:**

1. **Wait for indexing:**
   - Check bottom status bar for "Indexing..." or "Scanning files..."
   - Let it complete (can take a few minutes on first open)

2. **Invalidate caches:**
   - **File → Invalidate Caches → Invalidate and Restart**

3. **Check interpreter:**
   - Make sure `.venv` interpreter is selected
   - **File → Settings → Project → Python Interpreter**

4. **Reinstall stubs:**
   - **File → Settings → Project → Python Interpreter**
   - Click ⚙️ → **Show All**
   - Select your interpreter → Click **Show paths for the selected interpreter** (folder icon)
   - Click **Reload list of paths**

### Issue: Can't import from other Python files in project

**Cause:** Project root not marked as source root

**Solution:**

1. Right-click project root folder (`mlcon-berlin-2025`)
2. **Mark Directory as → Sources Root**
3. PyCharm will add it to PYTHONPATH
4. Restart run configuration

**Alternative:**
- **File → Settings → Project Structure**
- Mark `mlcon-berlin-2025` as **Sources**
- Click **OK**

---

## PyCharm Tips & Tricks

### 1. Quick Documentation

- **Hover** over any function/class to see documentation
- Or press `F1` (macOS) / `Ctrl+Q` (Windows/Linux)

### 2. Navigate to Definition

- `Cmd+Click` (macOS) or `Ctrl+Click` (Windows/Linux) on any symbol
- Or press `Cmd+B` / `Ctrl+B`

### 3. Find Usage

- Right-click symbol → **Find Usages**
- Or press `Alt+F7`

### 4. Refactor

- Right-click → **Refactor** → Choose option
- Or press `Ctrl+T` to see refactor menu
- Rename: `Shift+F6`

### 5. Format Code

- **Code → Reformat Code**
- Or press `Cmd+Alt+L` (macOS) / `Ctrl+Alt+L` (Windows/Linux)

### 6. Optimize Imports

- **Code → Optimize Imports**
- Or press `Ctrl+Alt+O`
- Removes unused imports, sorts them

### 7. Live Templates

Type abbreviation and press `Tab`:
- `main` → Creates `if __name__ == "__main__":` block
- `for` → Creates for loop
- `ifmain` → Creates main block

### 8. Multi-line Editing

- Hold `Alt` (Windows/Linux) or `Option` (macOS) and click multiple locations
- Or `Alt+Shift+Click` and drag for column selection

### 9. Run with Arguments

1. **Run → Edit Configurations**
2. Add **Script parameters** field
3. Run with arguments: `python script.py arg1 arg2`

### 10. Environment Variables

1. **Run → Edit Configurations**
2. Click **Environment variables** field
3. Add variables (e.g., API keys, tokens)
4. Format: `KEY=value` or use the dialog

---

## Keyboard Shortcuts Reference

| Action | macOS | Windows/Linux |
|--------|-------|---------------|
| **Run** | `Ctrl+Shift+R` | `Shift+F10` |
| **Debug** | `Ctrl+Shift+D` | `Shift+F9` |
| **Settings** | `Cmd+,` | `Ctrl+Alt+S` |
| **Terminal** | `Alt+F12` | `Alt+F12` |
| **Run Configurations** | `Ctrl+Alt+R` | `Shift+Alt+F10` |
| **Quick Documentation** | `F1` | `Ctrl+Q` |
| **Go to Definition** | `Cmd+B` | `Ctrl+B` |
| **Find Usages** | `Alt+F7` | `Alt+F7` |
| **Format Code** | `Cmd+Alt+L` | `Ctrl+Alt+L` |
| **Optimize Imports** | `Ctrl+Alt+O` | `Ctrl+Alt+O` |
| **Search Everywhere** | `Shift+Shift` | `Shift+Shift` |
| **Recent Files** | `Cmd+E` | `Ctrl+E` |
| **Navigate Back** | `Cmd+[` | `Ctrl+Alt+Left` |
| **Navigate Forward** | `Cmd+]` | `Ctrl+Alt+Right` |

---

## Advanced Configuration

### Custom Run Configuration Template

Create a template for all Python scripts:

1. **Run → Edit Configurations**
2. Click **Edit configuration templates** (bottom-left)
3. Select **Python**
4. Set default values:
   - **Working directory**: `$ProjectFileDir$`
   - **Add content roots to PYTHONPATH**: Checked
   - **Add source roots to PYTHONPATH**: Checked
5. Click **OK**

Now all new run configurations will use these settings.

### Enable Type Checking

PyCharm supports type hints and can check them:

1. **File → Settings → Editor → Inspections**
2. Navigate to **Python → Type Checker**
3. Enable **"Type checker compatibility inspection"**
4. Set severity to **Warning** or **Error**
5. Click **OK**

### Configure Code Style

Set up automatic formatting:

1. **File → Settings → Editor → Code Style → Python**
2. Configure indentation, spacing, imports, etc.
3. Or click **Set from...** → **PEP 8** for standard Python style
4. Enable **"Reformat code"** in commit dialog for auto-formatting

### Enable Auto-Import

Automatically add imports when using undefined symbols:

1. **File → Settings → Editor → General → Auto Import**
2. Check **"Show import popup"**
3. Check **"Optimize imports on the fly"**
4. Click **OK**

---

## Next Steps

Once PyCharm is configured:

1. **Explore the codebase:**
   - Use `Cmd+Shift+F` / `Ctrl+Shift+F` to search project
   - Use `Cmd+O` / `Ctrl+N` to quickly open files

2. **Run examples:**
   - Start with `day-1/ai_astrology_groq.py`
   - Try debugging to understand the flow

3. **Experiment:**
   - Modify prompts and parameters
   - Add breakpoints to inspect behavior

4. **Build your own:**
   - Create new Python files
   - Use the configured `.venv` automatically

Happy coding with PyCharm! 🚀
