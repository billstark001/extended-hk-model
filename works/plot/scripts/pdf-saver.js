/**
 * PDF保存模块
 * 用于将页面内容保存为PDF
 */
class PDFSaver {
    constructor() {
        this.isLoading = false;
    }
    
    async saveElementToPDF(elementId, filename = 'visualization.pdf') {
        if (this.isLoading) {
            console.warn('PDF保存正在进行中，请稍候...');
            return;
        }
        
        const element = document.getElementById(elementId);
        if (!element) {
            console.error('找不到要保存的元素:', elementId);
            return;
        }
        
        this.isLoading = true;
        const button = document.querySelector('.save-btn');
        const originalText = button ? button.textContent : '';
        
        try {
            // 更新按钮状态
            if (button) {
                button.textContent = '保存中...';
                button.disabled = true;
            }
            
            // 使用html2canvas截图
            const canvas = await html2canvas(element, {
                scale: 2,
                useCORS: true,
                allowTaint: true,
                backgroundColor: '#f4f4f9',
                logging: false
            });
            
            // 创建PDF
            const { jsPDF } = window.jspdf;
            const pdf = new jsPDF({
                orientation: 'landscape',
                unit: 'mm',
                format: 'a4'
            });
            
            const imgWidth = 297; // A4 landscape width in mm
            const imgHeight = (canvas.height * imgWidth) / canvas.width;
            
            // 如果图片高度超过A4纸张，需要分页
            if (imgHeight > 210) { // A4 landscape height is 210mm
                const pageHeight = 210;
                let position = 0;
                
                while (position < imgHeight) {
                    const remainingHeight = imgHeight - position;
                    const currentPageHeight = Math.min(pageHeight, remainingHeight);
                    
                    if (position > 0) {
                        pdf.addPage();
                    }
                    
                    pdf.addImage(
                        canvas.toDataURL('image/png'), 
                        'PNG', 
                        0, 
                        -position * (210 / imgHeight) * (imgHeight / pageHeight), 
                        imgWidth, 
                        imgHeight
                    );
                    
                    position += currentPageHeight;
                }
            } else {
                pdf.addImage(canvas.toDataURL('image/png'), 'PNG', 0, 0, imgWidth, imgHeight);
            }
            
            pdf.save(filename);
            
        } catch (error) {
            console.error('保存PDF时出错:', error);
            alert('保存PDF失败，请检查浏览器控制台获取详细错误信息。');
        } finally {
            // 恢复按钮状态
            if (button) {
                button.textContent = originalText;
                button.disabled = false;
            }
            this.isLoading = false;
        }
    }
}

// 导出为全局变量（为了保持零构建）
window.PDFSaver = PDFSaver;