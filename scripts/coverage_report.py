#!/usr/bin/env python3
"""
Comprehensive test coverage reporting script for DocFoundry.

This script provides detailed coverage analysis, generates reports,
and can be integrated into CI/CD pipelines.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional


class CoverageReporter:
    """Handles test coverage reporting and analysis."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_dir = project_root / "htmlcov"
        self.coverage_xml = project_root / "coverage.xml"
        self.coverage_json = project_root / "coverage.json"
        
    def run_tests_with_coverage(self, test_path: Optional[str] = None, 
                               fail_under: int = 75) -> bool:
        """Run tests with coverage reporting."""
        print("🧪 Running tests with coverage...")
        
        cmd = [
            "python", "-m", "pytest",
            test_path or "tests/",
            "--cov=.",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--cov-report=term-missing",
            f"--cov-fail-under={fail_under}",
            "-v"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, 
                                  capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def analyze_coverage(self) -> Dict:
        """Analyze coverage data and provide insights."""
        if not self.coverage_json.exists():
            print("❌ Coverage JSON file not found. Run tests first.")
            return {}
            
        try:
            with open(self.coverage_json, 'r') as f:
                coverage_data = json.load(f)
                
            analysis = {
                'total_coverage': coverage_data.get('totals', {}).get('percent_covered', 0),
                'total_lines': coverage_data.get('totals', {}).get('num_statements', 0),
                'covered_lines': coverage_data.get('totals', {}).get('covered_lines', 0),
                'missing_lines': coverage_data.get('totals', {}).get('missing_lines', 0),
                'files': {}
            }
            
            # Analyze per-file coverage
            files_data = coverage_data.get('files', {})
            for file_path, file_data in files_data.items():
                summary = file_data.get('summary', {})
                analysis['files'][file_path] = {
                    'coverage': summary.get('percent_covered', 0),
                    'lines': summary.get('num_statements', 0),
                    'covered': summary.get('covered_lines', 0),
                    'missing': summary.get('missing_lines', 0)
                }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing coverage: {e}")
            return {}
    
    def generate_summary_report(self, analysis: Dict) -> str:
        """Generate a human-readable coverage summary."""
        if not analysis:
            return "No coverage data available."
            
        total_coverage = analysis.get('total_coverage', 0)
        total_lines = analysis.get('total_lines', 0)
        covered_lines = analysis.get('covered_lines', 0)
        
        # Coverage status emoji
        if total_coverage >= 90:
            status = "🟢 Excellent"
        elif total_coverage >= 80:
            status = "🟡 Good"
        elif total_coverage >= 70:
            status = "🟠 Fair"
        else:
            status = "🔴 Needs Improvement"
            
        report = f"""
📊 COVERAGE SUMMARY
==================

Overall Coverage: {total_coverage:.1f}% {status}
Total Lines: {total_lines:,}
Covered Lines: {covered_lines:,}
Missing Lines: {total_lines - covered_lines:,}

📁 FILE COVERAGE BREAKDOWN
=========================
"""
        
        # Sort files by coverage (lowest first)
        files = analysis.get('files', {})
        sorted_files = sorted(files.items(), key=lambda x: x[1]['coverage'])
        
        for file_path, file_data in sorted_files:
            coverage = file_data['coverage']
            lines = file_data['lines']
            
            # Skip files with no executable lines
            if lines == 0:
                continue
                
            # Coverage indicator
            if coverage >= 90:
                indicator = "🟢"
            elif coverage >= 80:
                indicator = "🟡"
            elif coverage >= 70:
                indicator = "🟠"
            else:
                indicator = "🔴"
                
            # Shorten file path for readability
            short_path = file_path.replace(str(self.project_root), '').lstrip('/')
            if len(short_path) > 50:
                short_path = '...' + short_path[-47:]
                
            report += f"{indicator} {coverage:5.1f}% | {lines:4d} lines | {short_path}\n"
        
        return report
    
    def identify_uncovered_areas(self, analysis: Dict, threshold: float = 80.0) -> List[str]:
        """Identify files and areas that need more test coverage."""
        recommendations = []
        
        files = analysis.get('files', {})
        for file_path, file_data in files.items():
            coverage = file_data['coverage']
            lines = file_data['lines']
            
            # Skip files with no executable lines
            if lines == 0:
                continue
                
            if coverage < threshold:
                short_path = file_path.replace(str(self.project_root), '').lstrip('/')
                recommendations.append(
                    f"📝 {short_path}: {coverage:.1f}% coverage ({lines} lines)"
                )
        
        return recommendations
    
    def generate_badge_data(self, coverage: float) -> Dict:
        """Generate badge data for coverage reporting."""
        if coverage >= 90:
            color = "brightgreen"
        elif coverage >= 80:
            color = "green"
        elif coverage >= 70:
            color = "yellow"
        elif coverage >= 60:
            color = "orange"
        else:
            color = "red"
            
        return {
            "schemaVersion": 1,
            "label": "coverage",
            "message": f"{coverage:.1f}%",
            "color": color
        }


def main():
    parser = argparse.ArgumentParser(description="DocFoundry Coverage Reporter")
    parser.add_argument("--test-path", help="Specific test path to run")
    parser.add_argument("--fail-under", type=int, default=75, 
                       help="Minimum coverage percentage required")
    parser.add_argument("--threshold", type=float, default=80.0,
                       help="Coverage threshold for recommendations")
    parser.add_argument("--no-tests", action="store_true",
                       help="Skip running tests, analyze existing coverage")
    parser.add_argument("--badge", action="store_true",
                       help="Generate coverage badge data")
    
    args = parser.parse_args()
    
    # Find project root
    current_dir = Path.cwd()
    project_root = current_dir
    
    # Look for project markers
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "requirements.txt").exists() or (parent / "pyproject.toml").exists():
            project_root = parent
            break
    
    reporter = CoverageReporter(project_root)
    
    print(f"📁 Project root: {project_root}")
    
    # Run tests unless skipped
    if not args.no_tests:
        success = reporter.run_tests_with_coverage(args.test_path, args.fail_under)
        if not success:
            print("❌ Tests failed or coverage below threshold")
            sys.exit(1)
    
    # Analyze coverage
    analysis = reporter.analyze_coverage()
    if not analysis:
        print("❌ No coverage data to analyze")
        sys.exit(1)
    
    # Generate and display summary
    summary = reporter.generate_summary_report(analysis)
    print(summary)
    
    # Show recommendations
    recommendations = reporter.identify_uncovered_areas(analysis, args.threshold)
    if recommendations:
        print("\n🎯 COVERAGE IMPROVEMENT RECOMMENDATIONS")
        print("======================================")
        for rec in recommendations[:10]:  # Show top 10
            print(rec)
        
        if len(recommendations) > 10:
            print(f"... and {len(recommendations) - 10} more files")
    else:
        print(f"\n✅ All files meet the {args.threshold}% coverage threshold!")
    
    # Generate badge data if requested
    if args.badge:
        badge_data = reporter.generate_badge_data(analysis['total_coverage'])
        badge_file = project_root / "coverage-badge.json"
        with open(badge_file, 'w') as f:
            json.dump(badge_data, f, indent=2)
        print(f"\n🏷️  Coverage badge data saved to {badge_file}")
    
    # Final status
    total_coverage = analysis['total_coverage']
    if total_coverage >= args.fail_under:
        print(f"\n✅ Coverage check passed: {total_coverage:.1f}% >= {args.fail_under}%")
    else:
        print(f"\n❌ Coverage check failed: {total_coverage:.1f}% < {args.fail_under}%")
        sys.exit(1)


if __name__ == "__main__":
    main()