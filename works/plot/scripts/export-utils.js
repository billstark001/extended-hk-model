/**
 * 统一的导出工具模块
 * 支持PDF和SVG导出功能
 * 
 * 使用方法:
 * const exporter = new ExportUtils();
 * exporter.exportToPDF('main-container', 'my-file.pdf');
 * exporter.exportToSVG('mainSVG', 'my-file.svg');
 */
class ExportUtils {
    constructor() {
        this.isLoading = false;
        this.loadingButtons = new Set();
    }

    /**
     * 设置按钮加载状态
     * @param {string} buttonSelector - 按钮选择器
     * @param {boolean} loading - 是否加载中
     * @param {string} loadingText - 加载中显示的文本
     */
    setButtonLoading(buttonSelector, loading, loadingText = '处理中...') {
        const button = document.querySelector(buttonSelector);
        if (!button) return;

        if (loading) {
            if (!button.dataset.originalText) {
                button.dataset.originalText = button.textContent;
            }
            button.textContent = loadingText;
            button.disabled = true;
            this.loadingButtons.add(buttonSelector);
        } else {
            button.textContent = button.dataset.originalText || button.textContent;
            button.disabled = false;
            this.loadingButtons.delete(buttonSelector);
        }
    }

    /**
     * 导出DOM元素为PDF
     * @param {string} elementId - 要导出的元素ID
     * @param {string} filename - 文件名
     * @param {Object} options - 导出选项
     */
    async exportToPDF(elementId, filename = 'export.pdf', options = {}) {
        if (this.isLoading) {
            console.warn('PDF导出正在进行中，请稍候...');
            return;
        }

        const element = document.getElementById(elementId);
        if (!element) {
            console.error('找不到要导出的元素:', elementId);
            alert('找不到要导出的内容');
            return;
        }

        this.isLoading = true;
        this.setButtonLoading('.save-btn, .export-button, [onclick*="PDF"], [onclick*="pdf"]', true, '导出PDF中...');

        try {
            // 默认选项
            const defaultOptions = {
                scale: 2,
                useCORS: true,
                allowTaint: true,
                backgroundColor: '#ffffff',
                logging: false,
                orientation: 'landscape',
                format: 'a4',
                unit: 'mm',
                multiPage: true,
            };

            const finalOptions = { ...defaultOptions, ...options };

            // 使用html2canvas截图
            const canvas = await html2canvas(element, {
                scale: finalOptions.scale,
                useCORS: finalOptions.useCORS,
                allowTaint: finalOptions.allowTaint,
                backgroundColor: finalOptions.backgroundColor,
                logging: finalOptions.logging
            });

            // 创建PDF
            const { jsPDF } = window.jspdf;
            const pdf = new jsPDF({
                orientation: finalOptions.orientation,
                unit: finalOptions.unit,
                format: finalOptions.format
            });

            // 根据PDF格式和方向计算尺寸
            const pageWidth = finalOptions.format === 'a4' && finalOptions.orientation === 'landscape' ? 297 : 210;
            const pageHeight = finalOptions.format === 'a4' && finalOptions.orientation === 'landscape' ? 210 : 297;
            
            if (finalOptions.format === 'a3') {
                const a3Width = finalOptions.orientation === 'landscape' ? 420 : 297;
                const a3Height = finalOptions.orientation === 'landscape' ? 297 : 420;
                var imgWidth = a3Width;
                var maxHeight = a3Height;
            } else {
                var imgWidth = pageWidth;
                var maxHeight = pageHeight;
            }

            const imgHeight = (canvas.height * imgWidth) / canvas.width;

            // 如果图片高度超过页面，分页处理
            if (imgHeight > maxHeight && finalOptions.multiPage) {
                let position = 0;
                while (position < imgHeight) {
                    if (position > 0) {
                        pdf.addPage();
                    }
                    
                    const remainingHeight = imgHeight - position;
                    const currentHeight = Math.min(maxHeight, remainingHeight);
                    
                    pdf.addImage(
                        canvas.toDataURL('image/png'),
                        'PNG',
                        0,
                        -position,
                        imgWidth,
                        imgHeight
                    );
                    
                    position += maxHeight;
                }
            } else {
                pdf.addImage(canvas.toDataURL('image/png'), 'PNG', 0, (maxHeight - imgHeight) / 2, imgWidth, imgHeight);
            }

            pdf.save(filename);
            console.log(`PDF exported successfully: ${filename}`);

        } catch (error) {
            console.error('PDF导出失败:', error);
            alert('PDF导出失败，请查看控制台了解详细错误信息。');
        } finally {
            this.setButtonLoading('.save-btn, .export-button, [onclick*="PDF"], [onclick*="pdf"]', false);
            this.isLoading = false;
        }
    }

    /**
     * 导出SVG元素为SVG文件
     * @param {string} svgElementId - SVG元素ID
     * @param {string} filename - 文件名
     * @param {Object} options - 导出选项
     */
    async exportToSVG(svgElementId, filename = 'export.svg', options = {}) {
        if (this.isLoading) {
            console.warn('SVG导出正在进行中，请稍候...');
            return;
        }

        const svgElement = document.getElementById(svgElementId);
        if (!svgElement || svgElement.tagName.toLowerCase() !== 'svg') {
            console.error('找不到SVG元素或元素不是SVG:', svgElementId);
            alert('找不到要导出的SVG内容');
            return;
        }

        this.isLoading = true;
        this.setButtonLoading('[onclick*="SVG"], [onclick*="svg"]', true, '导出SVG中...');

        try {
            // 克隆SVG元素以避免修改原始元素
            const clonedSvg = svgElement.cloneNode(true);
            
            // 设置默认的命名空间和属性
            clonedSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
            clonedSvg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
            
            // 如果没有viewBox，根据width和height设置
            if (!clonedSvg.getAttribute('viewBox')) {
                const width = clonedSvg.getAttribute('width') || svgElement.getBoundingClientRect().width;
                const height = clonedSvg.getAttribute('height') || svgElement.getBoundingClientRect().height;
                clonedSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
            }

            // 应用内联样式（将CSS样式转换为内联样式）
            this.inlineStyles(clonedSvg);

            // 创建完整的SVG字符串
            const svgString = new XMLSerializer().serializeToString(clonedSvg);
            const fullSvgString = `<?xml version="1.0" encoding="UTF-8"?>\n${svgString}`;

            // 创建下载链接
            const blob = new Blob([fullSvgString], { type: 'image/svg+xml;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            // 清理URL对象
            URL.revokeObjectURL(url);
            
            console.log(`SVG exported successfully: ${filename}`);

        } catch (error) {
            console.error('SVG导出失败:', error);
            alert('SVG导出失败，请查看控制台了解详细错误信息。');
        } finally {
            this.setButtonLoading('[onclick*="SVG"], [onclick*="svg"]', false);
            this.isLoading = false;
        }
    }

    /**
     * 将外部CSS样式转换为内联样式
     * @param {Element} element - 要处理的元素
     */
    inlineStyles(element) {
        const computedStyle = window.getComputedStyle(element);
        const styleString = Array.from(computedStyle).reduce((str, property) => {
            return `${str}${property}:${computedStyle.getPropertyValue(property)};`;
        }, '');
        
        element.setAttribute('style', styleString);

        // 递归处理子元素
        Array.from(element.children).forEach(child => {
            this.inlineStyles(child);
        });
    }

    /**
     * 导出SVG为PDF（通过svg2pdf.js）
     * @param {string} svgElementId - SVG元素ID
     * @param {string} filename - 文件名
     * @param {Object} options - 导出选项
     */
    async exportSVGToPDF(svgElementId, filename = 'export.pdf', options = {}) {
        if (this.isLoading) {
            console.warn('SVG转PDF正在进行中，请稍候...');
            return;
        }

        const svgElement = document.getElementById(svgElementId);
        if (!svgElement || svgElement.tagName.toLowerCase() !== 'svg') {
            console.error('找不到SVG元素或元素不是SVG:', svgElementId);
            alert('找不到要导出的SVG内容');
            return;
        }

        // 检查svg2pdf.js是否可用
        if (typeof window.svg2pdf === 'undefined') {
            console.warn('svg2pdf.js not available, falling back to html2canvas method');
            return this.exportToPDF(svgElementId, filename, options);
        }

        this.isLoading = true;
        this.setButtonLoading('[onclick*="PDF"], [onclick*="pdf"]', true, '导出PDF中...');

        try {
            const { jsPDF } = window.jspdf;
            const defaultOptions = {
                orientation: 'landscape',
                unit: 'pt',
                format: [600, 450] // 默认SVG尺寸
            };

            const finalOptions = { ...defaultOptions, ...options };
            const pdf = new jsPDF(finalOptions);

            await window.svg2pdf(svgElement, pdf, {
                xOffset: 0,
                yOffset: 0,
                scale: 1
            });

            pdf.save(filename);
            console.log(`SVG to PDF exported successfully: ${filename}`);

        } catch (error) {
            console.error('SVG转PDF失败:', error);
            alert('SVG转PDF失败，请查看控制台了解详细错误信息。');
        } finally {
            this.setButtonLoading('[onclick*="PDF"], [onclick*="pdf"]', false);
            this.isLoading = false;
        }
    }

    /**
     * 批量导出多种格式
     * @param {string} elementId - 要导出的元素ID
     * @param {string} baseName - 基础文件名（不含扩展名）
     * @param {Array} formats - 要导出的格式数组 ['pdf', 'svg']
     * @param {Object} options - 导出选项
     */
    async exportMultipleFormats(elementId, baseName = 'export', formats = ['pdf', 'svg'], options = {}) {
        if (this.isLoading) {
            console.warn('批量导出正在进行中，请稍候...');
            return;
        }

        for (const format of formats) {
            try {
                switch (format.toLowerCase()) {
                    case 'pdf':
                        await this.exportToPDF(elementId, `${baseName}.pdf`, options.pdf || {});
                        break;
                    case 'svg':
                        await this.exportToSVG(elementId, `${baseName}.svg`, options.svg || {});
                        break;
                    case 'svgpdf':
                        await this.exportSVGToPDF(elementId, `${baseName}_vector.pdf`, options.svgpdf || {});
                        break;
                    default:
                        console.warn(`Unsupported format: ${format}`);
                }
                
                // 在导出之间稍作延迟
                await new Promise(resolve => setTimeout(resolve, 500));
            } catch (error) {
                console.error(`Error exporting ${format}:`, error);
            }
        }
    }

    /**
     * 检查必要的依赖库是否可用
     */
    checkDependencies() {
        const dependencies = {
            jsPDF: typeof window.jspdf !== 'undefined',
            html2canvas: typeof window.html2canvas !== 'undefined',
            svg2pdf: typeof window.svg2pdf !== 'undefined'
        };

        console.log('Export dependencies status:', dependencies);
        return dependencies;
    }
}

// 导出为全局变量（为了保持零构建兼容性）
window.ExportUtils = ExportUtils;

// 创建全局实例
window.exportUtils = new ExportUtils();