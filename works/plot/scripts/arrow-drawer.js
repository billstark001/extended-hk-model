/**
 * 箭头绘制模块
 * 用于在图表之间绘制连接箭头
 */
class ArrowDrawer {
    constructor(containerId, arrowsSvgId) {
        this.container = document.getElementById(containerId);
        this.arrowsSvg = document.getElementById(arrowsSvgId);
        this.setupSvg();
    }
    
    setupSvg() {
        if (!this.container || !this.arrowsSvg) {
            console.error('Container or arrows SVG not found');
            return;
        }
        
        const containerRect = this.container.getBoundingClientRect();
        this.arrowsSvg.setAttribute('width', containerRect.width);
        this.arrowsSvg.setAttribute('height', containerRect.height);
        this.arrowsSvg.style.position = 'absolute';
        this.arrowsSvg.style.top = '0';
        this.arrowsSvg.style.left = '0';
        this.arrowsSvg.style.width = '100%';
        this.arrowsSvg.style.height = '100%';
    }
    
    getElementCenter(selector) {
        const element = document.querySelector(selector);
        if (!element) return null;
        
        const rect = element.getBoundingClientRect();
        const containerRect = this.container.getBoundingClientRect();
        return {
            x: rect.left + rect.width / 2 - containerRect.left,
            y: rect.top + rect.height / 2 - containerRect.top
        };
    }
    
    drawArrow(fromSelector, toSelector, isDashed = false, offset = 60) {
        const fromCenter = this.getElementCenter(fromSelector);
        const toCenter = this.getElementCenter(toSelector);
        
        if (!fromCenter || !toCenter) {
            console.error('Could not find centers for', fromSelector, 'or', toSelector);
            return;
        }
        
        // 计算箭头方向和缩短距离
        const dx = toCenter.x - fromCenter.x;
        const dy = toCenter.y - fromCenter.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const unitX = dx / length;
        const unitY = dy / length;
        
        const startX = fromCenter.x + unitX * offset;
        const startY = fromCenter.y + unitY * offset;
        const endX = toCenter.x - unitX * offset;
        const endY = toCenter.y - unitY * offset;
        
        // 创建箭头线
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', startX);
        line.setAttribute('y1', startY);
        line.setAttribute('x2', endX);
        line.setAttribute('y2', endY);
        line.setAttribute('class', isDashed ? 'arrow-line dashed' : 'arrow-line');
        line.setAttribute('marker-end', isDashed ? 'url(#arrowhead-dashed)' : 'url(#arrowhead)');
        
        this.arrowsSvg.appendChild(line);
    }
    
    drawConnections(connections, offset) {
        // 清除现有箭头
        const existingLines = this.arrowsSvg.querySelectorAll('line');
        existingLines.forEach(line => line.remove());
        
        // 重新设置SVG尺寸
        this.setupSvg();
        
        // 绘制所有连接
        connections.forEach(([from, to, dashed]) => {
            this.drawArrow(from, to, dashed, offset);
        });
    }
}

// 导出为全局变量（为了保持零构建）
window.ArrowDrawer = ArrowDrawer;