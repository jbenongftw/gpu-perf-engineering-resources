# Contributing to GPU Performance Engineering Guide

Thank you for your interest in contributing! This guide aims to be the most comprehensive resource for GPU kernel programming in AI infrastructure.

## Quality Criteria

All submissions must meet these criteria:

### Must Have
- **Primary sources preferred** - Papers, official documentation, or direct practitioner experience
- **Real implementation insights** - Not just conceptual overviews
- **Verified accuracy** - Links must work, claims must be accurate
- **Unique value** - Resource adds something not already covered

### Red Flags (Likely Rejection)
- Surface-level "intro to CUDA" tutorials
- AI-generated content without significant human verification
- Marketing content disguised as technical resources
- Outdated resources (unless historically significant)
- Broken links or paywalled content without clear notation

## How to Contribute

### Adding a Resource

1. **Fork the repository**
2. **Determine the appropriate section and tier:**
   - **Tier 1**: Essential starting points everyone should know
   - **Tier 2**: Important deep dives and advanced topics
   - **Tier 3**: Specialized or cutting-edge material

3. **Use the correct format:**
   ```markdown
   📝 **Title** - Author/Source
   - [link.com](https://link.com)
   - Brief description (1-2 lines, focus on unique value)
   ```

4. **Add appropriate tags:**
   - `🔥` - Community favorite / highly recommended
   - `✨` - Recently added (within last 6 months)
   - `📄` - Academic paper
   - `📝` - Blog post or tutorial
   - `🎥` - Video content
   - `📚` - Book
   - `💻` - Code repository

5. **Submit a pull request** with:
   - Clear title: "Add [Resource Name] to [Section]"
   - Brief justification for why this resource is valuable
   - Your assessment of which tier it belongs in

### Updating Existing Resources

- Fix broken links
- Update descriptions if resources have changed significantly
- Suggest tier changes with justification

### Reporting Issues

- Broken links
- Outdated information
- Resources that no longer meet quality criteria
- Missing important resources in a category

## Section-Specific Guidelines

### Fundamentals
- Resources should be accessible to someone with basic programming knowledge
- Prefer timeless concepts over version-specific details

### Matrix Multiplication
- Focus on optimization techniques that generalize
- Include performance numbers when available

### Tensor Cores & Mixed Precision
- Keep up with latest architecture generations
- Note hardware requirements clearly

### Attention & Memory-Bound Kernels
- Paper citations should include arxiv links
- Note which architectures implementations target

### Compiler & DSL Approaches
- Include version/date information when relevant
- Note API stability if applicable

### Profiling & Optimization
- Practical, actionable guidance preferred
- Include tool versions when relevant

### AMD & Alternative Hardware
- Resources should be substantive, not just "we support X too"
- Note specific hardware requirements

### Production Inference Systems
- Focus on architectural insights, not just "how to deploy"
- Include performance characteristics

### LLM-Generated Kernels
- Note any caveats about benchmark gaming or limitations
- Include reproducibility information

### Distributed & Multi-GPU
- Scale of validation matters (tested on N GPUs)
- Note communication patterns and bottlenecks

## Review Process

1. Maintainers will review PRs within 1 week
2. Discussion may occur in PR comments
3. We may suggest alternative placement or tier
4. Merged contributions will include ✨ tag initially

## Code of Conduct

- Be respectful and constructive
- Focus on content quality, not author credentials
- Acknowledge that "best" resources vary by learning style
- Conflicts of interest (e.g., adding your own work) should be disclosed

## Questions?

Open an issue with the "question" label.
