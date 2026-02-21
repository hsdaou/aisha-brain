#!/usr/bin/env python3
"""Convenience wrapper — runs the knowledge builder from the package module."""
import sys
from aisha_brain.build_knowledge import scrape_and_save, build_index

if __name__ == '__main__':
    if '--scrape' in sys.argv:
        scrape_and_save()
    else:
        print('Rebuilding index from existing files (use --scrape to re-crawl website)')
    build_index()
