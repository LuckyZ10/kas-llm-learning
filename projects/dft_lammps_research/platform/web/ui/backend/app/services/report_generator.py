"""
Report Generator Service - Generate PDF/HTML reports
"""
from pathlib import Path
from typing import Optional
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.models.project import Project
from app.models.workflow import Workflow
from app.models.screening_result import ScreeningResult
from app.core.config import settings

logger = structlog.get_logger()


class ReportGenerator:
    """Generate research reports in various formats"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def generate_project_report(
        self,
        project_id: str,
        format: str = "pdf",
        include_structures: bool = True,
        include_charts: bool = True,
    ) -> Optional[Path]:
        """Generate comprehensive project report"""
        
        result = await self.db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        
        if not project:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"project_report_{project_id}_{timestamp}.{format}"
        output_path = settings.EXPORT_DIR / filename
        
        if format == "pdf":
            return await self._generate_pdf_report(project, output_path, include_structures, include_charts)
        elif format == "html":
            return await self._generate_html_report(project, output_path, include_structures, include_charts)
        elif format == "markdown":
            return await self._generate_markdown_report(project, output_path)
        
        return None
    
    async def _generate_pdf_report(
        self,
        project: Project,
        output_path: Path,
        include_structures: bool,
        include_charts: bool,
    ) -> Path:
        """Generate PDF report using ReportLab"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
        )
        story.append(Paragraph(f"Research Report: {project.name}", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Project info
        story.append(Paragraph("Project Information", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        info_data = [
            ["Property", "Value"],
            ["Name", project.name],
            ["Type", project.project_type],
            ["Material System", project.material_system or "N/A"],
            ["Status", project.status],
            ["Created", project.created_at.strftime("%Y-%m-%d %H:%M") if project.created_at else "N/A"],
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Statistics
        story.append(Paragraph("Statistics", styles['Heading2']))
        stats_data = [
            ["Metric", "Value"],
            ["Total Structures", str(project.total_structures)],
            ["Completed Calculations", str(project.completed_calculations)],
            ["Failed Calculations", str(project.failed_calculations)],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(stats_table)
        
        # Build PDF
        doc.build(story)
        logger.info(f"Generated PDF report", path=str(output_path))
        
        return output_path
    
    async def _generate_html_report(
        self,
        project: Project,
        output_path: Path,
        include_structures: bool,
        include_charts: bool,
    ) -> Path:
        """Generate HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Report: {project.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 32px; font-weight: bold; color: #3498db; }}
                .stat-label {{ color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <h1>Research Report: {project.name}</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Project Information</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Type</td><td>{project.project_type}</td></tr>
                <tr><td>Material System</td><td>{project.material_system or "N/A"}</td></tr>
                <tr><td>Status</td><td>{project.status}</td></tr>
                <tr><td>Created</td><td>{project.created_at.strftime("%Y-%m-%d %H:%M") if project.created_at else "N/A"}</td></tr>
            </table>
            
            <h2>Statistics</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{project.total_structures}</div>
                    <div class="stat-label">Total Structures</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{project.completed_calculations}</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{project.failed_calculations}</div>
                    <div class="stat-label">Failed</div>
                </div>
            </div>
        </body>
        </html>
        """
        
        output_path.write_text(html_content, encoding='utf-8')
        logger.info(f"Generated HTML report", path=str(output_path))
        
        return output_path
    
    async def _generate_markdown_report(self, project: Project, output_path: Path) -> Path:
        """Generate Markdown report"""
        
        markdown_content = f"""# Research Report: {project.name}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Project Information

| Property | Value |
|----------|-------|
| Name | {project.name} |
| Type | {project.project_type} |
| Material System | {project.material_system or "N/A"} |
| Status | {project.status} |
| Created | {project.created_at.strftime("%Y-%m-%d %H:%M") if project.created_at else "N/A"} |

## Statistics

- **Total Structures:** {project.total_structures}
- **Completed Calculations:** {project.completed_calculations}
- **Failed Calculations:** {project.failed_calculations}

## Description

{project.description or "No description provided."}
"""
        
        output_path.write_text(markdown_content, encoding='utf-8')
        logger.info(f"Generated Markdown report", path=str(output_path))
        
        return output_path
    
    async def generate_workflow_report(self, workflow_id: str, format: str = "pdf") -> Optional[Path]:
        """Generate workflow execution report"""
        result = await self.db.execute(select(Workflow).where(Workflow.id == workflow_id))
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_report_{workflow_id}_{timestamp}.{format}"
        output_path = settings.EXPORT_DIR / filename
        
        # Simplified implementation - would include task details, execution timeline, etc.
        if format == "html":
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Workflow Report: {workflow.name}</title></head>
            <body>
                <h1>Workflow Report: {workflow.name}</h1>
                <p>Status: {workflow.status}</p>
                <p>Progress: {workflow.progress_percent}%</p>
                <p>Type: {workflow.workflow_type}</p>
            </body>
            </html>
            """
            output_path.write_text(html_content)
            return output_path
        
        return None
    
    async def generate_screening_report(
        self,
        project_id: str,
        top_n: int = 50,
        format: str = "pdf",
    ) -> Optional[Path]:
        """Generate screening results report"""
        
        result = await self.db.execute(
            select(ScreeningResult)
            .where(ScreeningResult.project_id == project_id)
            .order_by(ScreeningResult.overall_score.desc())
            .limit(top_n)
        )
        results = result.scalars().all()
        
        if not results:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screening_report_{project_id}_{timestamp}.{format}"
        output_path = settings.EXPORT_DIR / filename
        
        if format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Structure ID', 'Formula', 'Formation Energy', 'Band Gap', 
                               'Ionic Conductivity', 'Overall Score'])
                for r in results:
                    writer.writerow([
                        r.structure_id, r.formula, r.formation_energy, r.band_gap,
                        r.ionic_conductivity, r.overall_score
                    ])
            return output_path
        
        elif format == "html":
            rows = ""
            for r in results:
                rows += f"<tr><td>{r.structure_id}</td><td>{r.formula}</td>"
                rows += f"<td>{r.formation_energy:.3f}</td><td>{r.band_gap:.3f}</td>"
                rows += f"<td>{r.ionic_conductivity:.3e}</td><td>{r.overall_score:.3f}</td></tr>"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Screening Results</title>
                <style>
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Screening Results - Top {top_n}</h1>
                <table>
                    <tr>
                        <th>Structure ID</th>
                        <th>Formula</th>
                        <th>Formation Energy (eV/atom)</th>
                        <th>Band Gap (eV)</th>
                        <th>Ionic Conductivity (S/cm)</th>
                        <th>Overall Score</th>
                    </tr>
                    {rows}
                </table>
            </body>
            </html>
            """
            output_path.write_text(html_content)
            return output_path
        
        return None
