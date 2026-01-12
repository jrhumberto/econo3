"""
Modulo de exportacao de resultados
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import streamlit as st


def export_to_pdf(results):
    try:
        from visualizations import (
            create_residuals_plot, create_qq_plot,
            create_prediction_plot, create_correlation_heatmap
        )
        
        buffer = BytesIO()
        
        with PdfPages(buffer) as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.95, 'Analise Econometrica', ha='center', fontsize=20, fontweight='bold')
            summary_text = str(results.get('summary', 'Resumo nao disponivel'))
            fig.text(0.1, 0.1, summary_text, fontsize=8, verticalalignment='top', family='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            if 'residuals' in results:
                fig = create_residuals_plot(results)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                fig = create_qq_plot(results)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                fig = create_prediction_plot(results)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            if 'data' in results:
                fig = create_correlation_heatmap(results['data'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {str(e)}")
        return None


def export_to_png(results):
    try:
        from visualizations import (
            create_residuals_plot, create_qq_plot,
            create_prediction_plot, create_correlation_heatmap
        )
        
        png_buffers = []
        
        if 'residuals' in results:
            fig = create_residuals_plot(results)
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            png_buffers.append(buffer)
            plt.close()
            
            fig = create_qq_plot(results)
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            png_buffers.append(buffer)
            plt.close()
            
            fig = create_prediction_plot(results)
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            png_buffers.append(buffer)
            plt.close()
        
        if 'data' in results:
            fig = create_correlation_heatmap(results['data'])
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            png_buffers.append(buffer)
            plt.close()
        
        return png_buffers
    
    except Exception as e:
        st.error(f"Erro ao gerar PNG: {str(e)}")
        return None
