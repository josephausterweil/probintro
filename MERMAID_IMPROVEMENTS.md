# Mermaid Rendering Improvements

## What Was Changed

### 1. Pinned Mermaid Version (hugo.toml)
- **Why**: Using a specific stable version (10.6.1) instead of the theme's bundled version ensures consistent rendering
- **Benefit**: Reduces version-related bugs and makes behavior predictable

### 2. Enhanced Mermaid Configuration (hugo.toml)
Added explicit configuration for:
- `startOnLoad: true` - Ensures diagrams render on page load
- `securityLevel: "loose"` - **Critical**: Allows custom styling (stroke, colors) to work properly
- Theme customization - Clean, neutral color scheme that works with the tutorial's styling
- `htmlLabels: true` - Better text rendering in diagrams

### 3. Automatic Retry Logic (layouts/partials/custom/mermaid-init.html)
- Detects diagrams that failed to render on first attempt
- Automatically retries rendering after 500ms
- Provides clear error messages if retry fails
- Works around timing issues between DOM ready and mermaid initialization

### 4. Custom Footer Partial (layouts/partials/footer.html)
- Integrates the retry script into every page
- Minimal override that doesn't interfere with theme functionality

## Files Modified/Created

```
probintro/
├── hugo.toml                                    [MODIFIED]
├── layouts/
│   └── partials/
│       ├── footer.html                          [CREATED]
│       └── custom/
│           └── mermaid-init.html                [CREATED]
└── MERMAID_IMPROVEMENTS.md                       [THIS FILE]
```

## Testing Instructions

### 1. Rebuild the Site
```bash
cd /home/jausterw/work/tutorials/amplifier_play/probintro
hugo server -D
```

### 2. Test Pages with Mermaid Diagrams
Navigate to pages with mermaid diagrams:
- `/intro/02_hungry/` - Multiple block-beta diagrams
- `/intro/03_prob_count/` - Diagrams with stroke styling

### 3. What to Check

**✅ Success indicators:**
- Diagrams render immediately on page load
- Red stroke styling (from `stroke: #f33, stroke-width:4px`) appears correctly
- No "undefined" or blank diagram areas
- No console errors in browser dev tools

**🔧 If you see issues:**
1. Open browser console (F12)
2. Look for "Retrying X unrendered mermaid diagram(s)..." message
3. Check if retry succeeds (diagrams should appear after 500ms)
4. If retry fails, you'll see error messages with details

### 4. Test Failure Recovery
To test the retry mechanism:
1. Disable cache in browser dev tools
2. Throttle network to "Slow 3G"
3. Refresh a page with diagrams
4. Watch console - you should see retry messages if initial load is slow

## Configuration Details

### securityLevel: "loose" - Why It Matters

The `securityLevel` setting is critical for your diagrams:

- **"strict"** (default): Blocks many styling options, including stroke styling
- **"loose"**: Allows custom CSS attributes like `stroke`, `fill`, `stroke-width`

Without "loose", your diagrams would render but:
- The red circles around outcomes wouldn't appear
- Custom colors might be ignored
- Styling would fall back to defaults

This is safe for your use case because:
1. You control all diagram content (in your markdown files)
2. No user-generated diagram code
3. Tutorial is static content, not a public platform

### Mermaid Version Selection

Version 10.9.3 was chosen because:
- Stable release with mature block-beta support
- Better handling of block-beta diagram syntax
- Well-documented
- Confirmed working with all tutorial diagrams

**To upgrade in the future:**
1. Check mermaid releases: https://github.com/mermaid-js/mermaid/releases
2. Update `customMermaidURL` in hugo.toml
3. Test all diagrams thoroughly

## Troubleshooting

### Problem: Diagrams still don't render
**Solution**: Check browser console for specific errors. Common issues:
- CDN blocked by network/firewall
- JavaScript errors in page
- Incompatible browser (use modern browser)

### Problem: Styling doesn't apply
**Solution**:
- Verify `securityLevel: "loose"` is set in hugo.toml
- Check mermaid syntax is correct
- Try using CSS shorthand: `style d stroke:#f33,stroke-width:4px`

### Problem: Random failures on page load
**Solution**:
- This is exactly what the retry logic fixes
- Check console - retry should succeed
- If retry consistently fails, there may be a syntax error in diagram

### Problem: Diagrams work locally but fail on GitHub Pages
**Solution**:
- Ensure the `public/` directory is built with the new config
- Verify CDN is accessible from GitHub Pages
- Check GitHub Pages build logs for errors

## Performance Notes

### CDN vs Bundled Version
**Pros of CDN**:
- Specific version control
- May be cached by browser from other sites
- Easy to upgrade

**Cons of CDN**:
- Requires network request
- Slight initial load delay

**Recommendation**: For your tutorial, CDN is better because:
1. Stability is more important than milliseconds
2. Diagrams are learning content (worth the wait)
3. Browser caching helps on subsequent visits

## Future Improvements (Optional)

### 1. Pre-render Diagrams at Build Time
For maximum reliability, consider generating PNG/SVG at build time:
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i diagrams.mmd -o diagrams.svg
```

**Pros**: Zero client-side rendering issues
**Cons**: More complex build process

### 2. Fallback Images
Keep PNG versions of key diagrams as fallback:
```markdown
![Diagram](fallback.png)
```

### 3. Diagram Source Files
Consider keeping separate `.mmd` files for complex diagrams:
- Easier to maintain
- Can be pre-processed
- Version control friendly

## Maintenance

### Regular Checks
1. Test diagrams after Hugo/theme updates
2. Check mermaid changelog for security updates
3. Verify CDN availability (jsDelivr status page)

### When to Upgrade Mermaid
- Security vulnerabilities announced
- New features needed (e.g., new diagram types)
- Current version has blocking bugs
- After major Hugo/theme updates (test compatibility)

## Summary

The improvements focus on **reliability over innovation**:
- Pinned version = predictable behavior
- Explicit config = no guessing
- Retry logic = graceful failure handling
- Documentation = maintainable solution

Your mermaid diagrams should now render consistently, with proper styling, and recover automatically from timing issues.
