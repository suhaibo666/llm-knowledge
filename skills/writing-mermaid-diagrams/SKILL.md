---
name: writing-mermaid-diagrams
description: Use when adding or editing a Mermaid diagram in this wiki - which characters break the parser, the two severity tiers, and the mandatory post-generation review. Load it before drawing a flowchart, sequence or state diagram.
---

# Writing Mermaid Diagrams

这些规则是本库实测沉淀：mermaid 渲染失败几乎全部来自「节点/连线文本里混进了 mermaid 自己的语法定界符」。新增或改写任何 mermaid 块前读一遍，写完按末尾的校验清单逐条过。
Mermaid 渲染失败源于**节点/连线文本里混入了 mermaid 的语法定界符**(`[] () {} |` 和换行)。原理:mermaid 用 `[]`/`()`/`{}` 界定节点形状、用 `|` 界定连线标签,文本里再出现这些字符可能让解析器判断不出边界。**按严重度分两档**(本库实测沉淀):

> **① 必崩(零容忍)—— 形状内嵌套定界符**:`X[(无 [N,V] 激活)]`(圆柱 `[(...)]` 里又有 `[`)、`{判断[i]}`(菱形里有 `[`)这类**特殊形状内再出现 `[]`/`()`** 一定解析失败。修法:特殊形状(圆柱 `[(...)]`、子程序 `[[...]]`、菱形 `{...}`)内**只放最简纯文本**,绝不嵌套定界符 → `X["无 N×V 激活"]`。
>
> **② 渲染器相关(求稳一律避免)**:`A["logits[N,V]"]`(带引号矩形标签里的 `[]`/`()`)、`A -. "文字" .-> B`(虚线内联引号标签)——这些在多数 mermaid 版本能渲、个别版本崩,**不稳定**。为可移植求稳:张量形状写 `N×d`/`B·S·V` 不写 `[N,V]`;带文字连线统一用管道标签 `A -->|文字| B` / `A -.->|文字| B`(文字内不放引号、括号、`|`)。已能渲的旧图不必为此大改,但新图按此写。

**其它常见坑 → 修法:**
- 管道标签里带引号/括号:`A -->|"文字(x)"| B` → `A -->|文字 x| B`。
- 标签里直接敲回车换行 → 用 `<br/>`。
- 中文/符号当节点 id → id 用英文数字(`H`、`NA`),中文/特殊字符放进 `["..."]` 标签里。
- 子图标题 `subgraph id["标题"]`:标题可含 `( )`、`：`、`/`(已验证可渲),但**不可含 `[ ]` 或 `|`**;每个 subgraph 必须用**单独一行** `end` 闭合。
- 代码块**首行**必须是图类型声明(`flowchart TB|LR`、`sequenceDiagram`、`graph TD` 等)。

**生成后校验(必做,不可跳):**
1. 写完一个 mermaid 块**立即重读**它,对照上面逐条扫:标签里有没有裸 `[] ()`、特殊形状有没有嵌套定界符、连线文字有没有引号/括号/`|`、有没有 `<br/>` 之外的换行。
2. 提交前对本次改动的每个文件 `grep -n mermaid <file>` 定位所有图,**逐块**再过一遍清单。
3. 有条件就实渲确认(mermaid-cli `mmdc`,或在线 live editor 粘一遍);不能渲就严格按清单人工核对。
4. 发现问题就地改;**绝不**把"可能能渲"的块留到 commit。
