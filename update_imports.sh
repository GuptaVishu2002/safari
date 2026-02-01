#!/bin/bash

# Script to update imports from src.* to safari.*
# Run this from the root of your safari repository AFTER renaming src to safari

echo "Finding all Python files with 'src.' imports..."
echo ""

# First, let's see what needs to be changed
echo "=== Files containing 'src.' imports ==="
grep -r "from src\." safari/ --include="*.py" -l 2>/dev/null || echo "No 'from src.' imports found"
grep -r "import src\." safari/ --include="*.py" -l 2>/dev/null || echo "No 'import src.' imports found"

echo ""
echo "=== Sample imports that will be changed ==="
grep -r "from src\." safari/ --include="*.py" -h 2>/dev/null | head -10
grep -r "import src\." safari/ --include="*.py" -h 2>/dev/null | head -10

echo ""
read -p "Do you want to proceed with replacing 'src.' with 'safari.' in all Python files? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Replacing imports..."
    
    # Find all Python files and replace src. with safari.
    find safari/ -name "*.py" -type f -exec sed -i 's/from src\./from safari./g' {} +
    find safari/ -name "*.py" -type f -exec sed -i 's/import src\./import safari./g' {} +
    
    # Also check standalone scripts in root
    for file in *.py; do
        if [ -f "$file" ]; then
            sed -i 's/from src\./from safari./g' "$file"
            sed -i 's/import src\./import safari./g' "$file"
        fi
    done
    
    echo ""
    echo "Done! All 'src.' imports have been replaced with 'safari.'"
    echo ""
    echo "=== Verification - showing updated imports ==="
    grep -r "from safari\." safari/ --include="*.py" -h 2>/dev/null | head -10
else
    echo "Aborted."
fi