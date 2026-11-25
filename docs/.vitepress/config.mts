import { defineConfig } from 'vitepress'
// https://vitepress.dev/reference/site-config

// 1. 获取环境变量并判断
// 如果环境变量 EDGEONE 等于 '1'，说明在 EdgeOne 环境，使用根路径 '/'
// 否则默认是 GitHub Pages 环境，使用仓库子路径 '/easy-vecdb/'
const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/repo-template/'

export default defineConfig({
  lang: 'zh-CN',
  title: "VibeKaggle",
  description: "用大模型打 Kaggle 的全新人机协作方式",
  base: baseConfig,
  markdown: {
    math: true
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/datawhale-logo.png',
    nav: [
      { text: 'PDF版本下载', link: 'https://github.com/datawhalechina/repo-template/releases' },
    ],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },
    sidebar: [
      {
        text: '前言：VibeKaggle 不是用 AI 写代码',
        link: '/chapters/preface.html',
      },
      {
        text: '第1部分：入门与基础',
        collapsible: true,
        items: [
          { text: '第1章：VibeKaggle 的工作流全景', link: '/chapter01/' },
          { text: '第2章：VibeKaggle 的操作系统', link: '/chapter02/' },
        ],
      },
      {
        text: '第2部分：表格数据：VibeKaggle 的主场',
        collapsible: true,
        items: [
          { text: '第3章：表格数据的 Vibe 工作流', link: '/chapter03/' },
          { text: '第4章：S6E3 客户流失预测（NVIDIA 夺冠）', link: '/chapter04/' },
          { text: '第5章：S5E10 道路事故风险预测', link: '/chapter05/' },
          { text: '第6章：S6E1 学生考试成绩预测', link: '/chapter06/' },
          { text: '第7章：NeurIPS Open Polymer Prediction 2025', link: '/chapter07/' },
        ],
      },
      {
        text: '第3部分：时序数据',
        collapsible: true,
        items: [
          { text: '第8章：时序数据的 Vibe 工作流', link: '/chapter08/' },
          { text: '第9章：S5E1 贴纸销售预测', link: '/chapter09/' },
          { text: '第10章：S5E10 道路事故风险预测', link: '/chapter10/' },
          { text: '第11章：Jane Street 实时市场数据预测', link: '/chapter11/' },
        ],
      },
      {
        text: '第4部分：图像数据',
        collapsible: true,
        items: [
          { text: '第12章：图像数据的 Vibe 工作流', link: '/chapter12/' },
          { text: '第13章：CSIRO Image2Biomass 预测', link: '/chapter13/' },
          { text: '第14章：RSNA Knee 异常检测', link: '/chapter14/' },
          { text: '第15章：Image Matching Challenge 2025', link: '/chapter15/' },
        ],
      },
      {
        text: '第5部分：文本数据',
        collapsible: true,
        items: [
          { text: '第16章：文本数据的 Vibe 工作流', link: '/chapter16/' },
          { text: '第17章：Jigsaw 社区规则分类', link: '/chapter17/' },
          { text: '第18章：LLM Prompt Recovery', link: '/chapter18/' },
          { text: '第19章：Make Data Count', link: '/chapter19/' },
        ],
      },
      {
        text: '第6部分：进阶与迁移',
        collapsible: true,
        items: [
          { text: '第20章：验证与评估', link: '/chapter20/' },
          { text: '第21章：调优与集成', link: '/chapter21/' },
          { text: '第22章：策略与心态', link: '/chapter22/' },
          { text: '第23章：从 VibeKaggle 到实际项目', link: '/chapter23/' },
        ],
      },
      {
        text: '附录',
        collapsible: true,
        items: [
          { text: '附录A：NumPy / pandas / matplotlib', link: '/appendices/appendix_a.html' },
          { text: '附录B：scikit-learn / XGBoost / LightGBM', link: '/appendices/appendix_b.html' },
          { text: '附录C：AutoGluon', link: '/appendices/appendix_c.html' },
          { text: '附录D：PyTorch / Transformers', link: '/appendices/appendix_d.html' },
          { text: '附录E：表格基础模型', link: '/appendices/appendix_e.html' },
          { text: '附录F：Kaggle MCP 配置', link: '/appendices/appendix_f.html' },
          { text: '附录G：评估指标速查', link: '/appendices/appendix_g.html' },
          { text: '附录H：工件管理与实验日志', link: '/appendices/appendix_h.html' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/repo-template' }
    ],

    editLink: {
      pattern: 'https://github.com/datawhalechina/repo-template/blob/main/docs/:path'
    },

    footer: {
      message: '<a href="https://beian.miit.gov.cn/" target="_blank">京ICP备2026002630号-1</a> | <a href="https://beian.mps.gov.cn/#/query/webSearch?code=11010602202215" rel="noreferrer" target="_blank">京公网安备11010602202215号</a>',
      copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）</a> 进行许可'
    }
  }
})
