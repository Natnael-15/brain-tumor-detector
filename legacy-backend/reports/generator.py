"""
Medical Report Generation Module

This module generates comprehensive medical reports for brain tumor analysis
including findings, statistics, and visualizations.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Try to import optional dependencies
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates medical-style reports for brain tumor analysis."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.styles = self._create_styles()
        
    def _create_styles(self) -> Dict:
        """Create custom styles for the report."""
        if REPORTLAB_AVAILABLE:
            styles = getSampleStyleSheet()
            
            # Custom styles
            styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=styles['Title'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center
            ))
            
            styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=styles['Heading1'],
                fontSize=14,
                textColor=colors.blue,
                spaceAfter=12
            ))
            
            styles.add(ParagraphStyle(
                name='Finding',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                leftIndent=20
            ))
            
            return styles
        else:
            return {}
    
    def generate_html_report(self, analysis_data: Dict, output_path: str):
        """
        Generate an HTML report for web viewing.
        
        Args:
            analysis_data: Dictionary containing analysis results
            output_path: Path to save the HTML report
        """
        html_content = self._create_html_content(analysis_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
    
    def _create_html_content(self, data: Dict) -> str:
        """Create HTML content for the report."""
        patient_info = data.get('patient_info', {})
        scan_info = data.get('scan_info', {})
        analysis_results = data.get('analysis_results', {})
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Brain MRI Tumor Analysis Report</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #2c3e50;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .hospital-name {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .report-title {{
                    font-size: 20px;
                    color: #34495e;
                    margin-top: 10px;
                }}
                .section {{
                    margin-bottom: 25px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #3498db;
                }}
                .section-title {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 10px;
                }}
                .info-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 15px;
                }}
                .info-table th, .info-table td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .info-table th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                .finding {{
                    background-color: #e8f4f8;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                .critical {{
                    background-color: #ffebee;
                    border-left: 4px solid #f44336;
                }}
                .normal {{
                    background-color: #e8f5e8;
                    border-left: 4px solid #4caf50;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 2px solid #2c3e50;
                    text-align: center;
                    font-size: 12px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="hospital-name">Medical Imaging Analysis Center</div>
                <div class="report-title">Brain MRI Tumor Analysis Report</div>
                <div>Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
            </div>
            
            <div class="section">
                <div class="section-title">Patient Information</div>
                <table class="info-table">
                    <tr><th>Patient ID</th><td>{patient_info.get('id', 'N/A')}</td></tr>
                    <tr><th>Age</th><td>{patient_info.get('age', 'N/A')}</td></tr>
                    <tr><th>Gender</th><td>{patient_info.get('gender', 'N/A')}</td></tr>
                    <tr><th>Study Date</th><td>{patient_info.get('study_date', 'N/A')}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <div class="section-title">Scan Information</div>
                <table class="info-table">
                    <tr><th>Scanner Type</th><td>{scan_info.get('scanner', 'N/A')}</td></tr>
                    <tr><th>Sequence</th><td>{scan_info.get('sequence', 'N/A')}</td></tr>
                    <tr><th>Field Strength</th><td>{scan_info.get('field_strength', 'N/A')}</td></tr>
                    <tr><th>Slice Thickness</th><td>{scan_info.get('slice_thickness', 'N/A')}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <div class="section-title">Analysis Results</div>
                {self._generate_findings_html(analysis_results)}
            </div>
            
            <div class="section">
                <div class="section-title">Tumor Volumetric Analysis</div>
                {self._generate_volumetrics_html(analysis_results)}
            </div>
            
            <div class="section">
                <div class="section-title">Analysis Model Information</div>
                <table class="info-table">
                    <tr><th>Model Type</th><td>{analysis_results.get('model_type', 'Deep Learning CNN')}</td></tr>
                    <tr><th>Confidence Score</th><td>{analysis_results.get('confidence', 0):.2%}</td></tr>
                    <tr><th>Processing Time</th><td>{analysis_results.get('processing_time', 'N/A')}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <div class="section-title">Recommendations</div>
                {self._generate_recommendations_html(analysis_results)}
            </div>
            
            <div class="footer">
                <p><strong>Disclaimer:</strong> This automated analysis report is intended for research and educational purposes only. 
                All clinical decisions must be made by qualified medical professionals after comprehensive review of patient history, 
                clinical presentation, and all available diagnostic data. This system provides decision support and should not be used 
                as the sole basis for diagnosis or treatment planning.</p>
                <p>Report generated by Brain MRI Tumor Detector v1.0</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_findings_html(self, results: Dict) -> str:
        """Generate HTML for findings section."""
        has_tumor = results.get('has_tumor', False)
        
        if has_tumor:
            tumor_type = results.get('predicted_label', 'Unknown')
            confidence = results.get('confidence', 0)
            
            finding_class = "critical" if confidence > 0.8 else "finding"
            
            html = f"""
            <div class="finding {finding_class}">
                <strong>FINDING:</strong> Tumor detected - {tumor_type}
                <br><strong>Confidence:</strong> {confidence:.1%}
                <br><strong>Location:</strong> {results.get('location', 'To be determined by radiologist')}
            </div>
            """
        else:
            html = """
            <div class="finding normal">
                <strong>FINDING:</strong> No significant abnormalities detected
                <br><strong>Status:</strong> Normal brain tissue appearance
            </div>
            """
        
        return html
    
    def _generate_volumetrics_html(self, results: Dict) -> str:
        """Generate HTML for volumetric analysis."""
        volumes = results.get('tumor_volumes', {})
        
        if not volumes:
            return "<p>No volumetric data available</p>"
        
        html = "<table class='info-table'>"
        html += "<tr><th>Tissue Type</th><th>Volume (mm³)</th><th>Percentage</th></tr>"
        
        total_volume = sum(volumes.values())
        
        for tissue_type, volume in volumes.items():
            if volume > 0:
                percentage = (volume / total_volume * 100) if total_volume > 0 else 0
                html += f"<tr><td>{tissue_type.title()}</td><td>{volume:,.0f}</td><td>{percentage:.1f}%</td></tr>"
        
        html += "</table>"
        
        return html
    
    def _generate_recommendations_html(self, results: Dict) -> str:
        """Generate HTML for recommendations section."""
        has_tumor = results.get('has_tumor', False)
        confidence = results.get('confidence', 0)
        
        if has_tumor and confidence > 0.7:
            recommendations = [
                "Immediate consultation with a neuro-oncologist recommended",
                "Consider additional imaging studies (contrast-enhanced MRI, perfusion imaging)",
                "Tissue sampling may be required for definitive diagnosis",
                "Multidisciplinary team consultation for treatment planning"
            ]
        elif has_tumor and confidence <= 0.7:
            recommendations = [
                "Follow-up imaging in 3-6 months recommended",
                "Clinical correlation with symptoms and physical examination",
                "Consider consultation with neurologist",
                "AI finding requires radiologist review and confirmation"
            ]
        else:
            recommendations = [
                "Routine follow-up as clinically indicated",
                "No immediate intervention required based on AI analysis",
                "Continue regular screening as per medical guidelines"
            ]
        
        html = "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        
        return html
    
    def generate_pdf_report(self, analysis_data: Dict, output_path: str):
        """
        Generate a PDF report using ReportLab.
        
        Args:
            analysis_data: Dictionary containing analysis results
            output_path: Path to save the PDF report
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. Generating text report instead.")
            self.generate_text_report(analysis_data, output_path.replace('.pdf', '.txt'))
            return
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph("Brain MRI Tumor Analysis Report", self.styles['ReportTitle'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Patient Information
        story.append(Paragraph("Patient Information", self.styles['SectionHeader']))
        patient_data = self._format_patient_data(analysis_data.get('patient_info', {}))
        story.append(patient_data)
        story.append(Spacer(1, 12))
        
        # Analysis Results
        story.append(Paragraph("Analysis Results", self.styles['SectionHeader']))
        results_text = self._format_results_text(analysis_data.get('analysis_results', {}))
        story.append(Paragraph(results_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")
    
    def generate_text_report(self, analysis_data: Dict, output_path: str):
        """
        Generate a simple text report.
        
        Args:
            analysis_data: Dictionary containing analysis results
            output_path: Path to save the text report
        """
        content = []
        content.append("=" * 60)
        content.append("BRAIN MRI TUMOR ANALYSIS REPORT")
        content.append("=" * 60)
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Patient Info
        content.append("PATIENT INFORMATION:")
        content.append("-" * 20)
        patient_info = analysis_data.get('patient_info', {})
        for key, value in patient_info.items():
            content.append(f"{key.title()}: {value}")
        content.append("")
        
        # Analysis Results
        content.append("ANALYSIS RESULTS:")
        content.append("-" * 17)
        results = analysis_data.get('analysis_results', {})
        
        has_tumor = results.get('has_tumor', False)
        if has_tumor:
            content.append(f"FINDING: Tumor detected")
            content.append(f"Type: {results.get('predicted_label', 'Unknown')}")
            content.append(f"Confidence: {results.get('confidence', 0):.1%}")
        else:
            content.append("FINDING: No significant abnormalities detected")
        
        content.append("")
        content.append("DISCLAIMER:")
        content.append("This automated analysis report is for research and educational purposes only.")
        content.append("All clinical decisions must be made by qualified medical professionals based on")
        content.append("comprehensive evaluation of patient history, symptoms, and additional diagnostic tests.")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(content))
        
        logger.info(f"Text report generated: {output_path}")
    
    def create_report(self, analysis_results: Dict, output_path: str, format_type: str = 'html'):
        """
        Main interface for creating reports.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_path: Path to save the report
            format_type: Type of report ('html', 'pdf', 'text')
        """
        # Add timestamp and metadata
        report_data = {
            'patient_info': analysis_results.get('patient_info', {
                'id': 'DEMO_001',
                'age': 'Unknown',
                'gender': 'Unknown',
                'study_date': datetime.now().strftime('%Y-%m-%d')
            }),
            'scan_info': analysis_results.get('scan_info', {
                'scanner': 'Demo Scanner',
                'sequence': 'T1-weighted',
                'field_strength': '3T',
                'slice_thickness': '1mm'
            }),
            'analysis_results': analysis_results,
            'timestamp': datetime.now().isoformat()
        }
        
        if format_type.lower() == 'html':
            self.generate_html_report(report_data, output_path)
        elif format_type.lower() == 'pdf':
            self.generate_pdf_report(report_data, output_path)
        elif format_type.lower() == 'text':
            self.generate_text_report(report_data, output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")


def main():
    """Command line interface for report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Report Generator")
    parser.add_argument("--results", required=True, help="Path to analysis results JSON")
    parser.add_argument("--output", required=True, help="Output report path")
    parser.add_argument("--format", choices=['html', 'pdf', 'text'], default='html', help="Report format")
    
    args = parser.parse_args()
    
    # Load analysis results
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Generate report
    generator = ReportGenerator()
    generator.create_report(results, args.output, args.format)


if __name__ == "__main__":
    main()