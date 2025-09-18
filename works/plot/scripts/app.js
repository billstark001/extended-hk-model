/**
 * 网络状态可视化应用主控制器
 */
class NetworkVisualizationApp {
    constructor() {
        this.arrowDrawer = null;
        this.connections = [
            ['#graph1', '#graph2', false], // 1-2 实线
            ['#graph2', '#graph4', false], // 2-4 实线  
            ['#graph1', '#graph3', false], // 1-3 实线
            ['#graph3', '#graph5', false], // 3-5 实线
            ['#graph2', '#graph5', true],  // 2-5 虚线
            ['#graph3', '#graph4', true]   // 3-4 虚线
        ];
    }
    
    init() {
        // 等待页面完全加载
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupApp());
        } else {
            this.setupApp();
        }
    }
    
    setupApp() {
        // 初始化组件
        this.arrowDrawer = new ArrowDrawer('main-container', 'arrows-svg');
        
        // 延迟绘制箭头，确保图表已完全渲染
        setTimeout(() => {
            this.drawArrows();
        }, 200);
        
        // 监听窗口大小变化
        window.addEventListener('resize', () => {
            setTimeout(() => this.drawArrows(), 100);
        });
    }
    
    drawArrows() {
        if (this.arrowDrawer) {
            this.arrowDrawer.drawConnections(this.connections, 180);
        }
    }
}

// 导出为全局变量（为了保持零构建）
window.NetworkVisualizationApp = NetworkVisualizationApp;