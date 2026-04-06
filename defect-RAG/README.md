# 🔍 软件缺陷分析RAG系统

基于历史缺陷数据的智能分析系统，支持相似缺陷检索、根因分析和改进建议。

## 功能特性

- 🔍 **相似缺陷检索**：基于语义向量检索历史相似缺陷
- 🧠 **根因分析**：AI驱动的根因推断和分类
- 💬 **对话界面**：支持多轮对话，中英双语
- 📁 **数据管理**：支持上传新数据，增量更新索引
- ⚙️ **参数可调**：temperature、max_tokens、top_k等参数实时调整

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API

编辑 `config/settings.yaml` 或设置环境变量：

```yaml
llm:
  base_url: "http://your-api-endpoint/v1"
  api_key: "your-api-key"
  model: "your-model-name"

embedding:
  base_url: "http://your-api-endpoint/v1"
  api_key: "your-api-key"
  model: "your-embedding-model"
```

### 3. 启动应用

```bash
streamlit run main.py
```

### 4. 构建索引

1. 在侧边栏配置API参数并保存
2. 上传缺陷数据JSON文件
3. 点击"构建索引"按钮

### 5. 开始分析

在对话界面输入缺陷描述，系统将返回：
- 最可能的根因
- 相似历史缺陷
- 预防和改进建议

## 数据结构

支持的JSON格式：

```json
{
  "Sheet1": [
    {
      "Identifier": 12345,
      "Summary": "缺陷标题",
      "PreClarification": "问题描述",
      "CategoryOfGaps": "缺陷类型",
      "Component": "组件",
      "Customer": "客户",
      ...
    }
  ]
}
```

或直接数组格式：

```json
[
  {
    "Identifier": 12345,
    "Summary": "缺陷标题",
    ...
  }
]
```

## 项目结构

```
defect-RAG/
├── config/              # 配置管理
├── src/
│   ├── core/           # 核心模块
│   │   ├── data_loader.py
│   │   ├── embedding_engine.py
│   │   ├── vector_store.py
│   │   ├── llm_client.py
│   │   └── index_manager.py
│   ├── chains/         # RAG链
│   │   ├── prompts.py
│   │   └── rag_chain.py
│   ├── ui/             # 界面
│   │   └── app.py
│   └── utils/          # 工具
├── vector_db/          # 向量数据库
├── uploads/            # 上传文件
├── main.py             # 入口文件
└── requirements.txt
```

## API兼容性

本系统使用OpenAI Compatible API格式，支持：
- vLLM
- Ollama
- 任何OpenAI API兼容的服务

## 参数说明

| 参数 | 说明 | 范围 |
|------|------|------|
| temperature | 创造性程度 | 0.0-2.0 |
| max_tokens | 最大输出长度 | 256-4096 |
| top_k | 检索相似缺陷数量 | 1-20 |

## 许可证

MIT License
