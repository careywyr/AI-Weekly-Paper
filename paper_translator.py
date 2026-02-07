#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HuggingFace论文抓取与翻译脚本
功能：
1. 从HuggingFace weekly paper页面抓取论文列表
2. 通过Arxiv客户端获取论文详细信息
3. 使用大模型翻译标题和摘要
"""

import requests
import os
import re
import arxiv
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
from typing import Optional, List, Dict

# 基础配置
BASE_URL = "https://huggingface.co"
LIKES_THRESHOLD = 45  # 点赞数阈值
TARGET_WEEK = "2026-W06"  # 目标周，格式如 "2026-W04"，None 表示使用当前周

# LLM配置
LLM_CONFIG = {
    "deepseek": {
        "model_name": "deepseek-chat",
        "api_key": os.environ.get("DEEPSEEK_KEY"),
        "base_url": "https://api.deepseek.com",
    }
}

# 翻译提示词
TRANSLATION_PROMPT = """
你是一位精通简体中文的专业翻译，尤其擅长将英文的专业学术论文或文章翻译成面向专业技术人员的中文技术文章。请你帮我将以下英文段落翻译成中文，风格与中文理工技术书籍读物相似。

规则：
- 翻译时要准确传达原文的事实和背景。
- 即使上意译也要保留原始段落格式，以及保留术语，例如 FLAC，JPEG 等。保留公司缩写，例如 Microsoft, Amazon, OpenAI 等。
- Markdown 标题（例如 "## Title"）请保持英文原文，不要翻译。
- Markdown 链接的文本（例如 "[Title](Link)" 中的 "Title"）需要翻译成中文。
- 人名不翻译
- 全角括号换成半角括号，并在左括号前面加半角空格，右括号后面加半角空格。
- 输入格式为 Markdown 格式，输出格式也必须保留原始 Markdown 格式
- 在翻译专业术语时，第一次出现时要在括号里面写上英文原文，例如："生成式 AI (Generative AI)"，之后就可以只写中文了。
- 注意你翻译内容的受众是专业技术人员，因此不需要对专业术语做口语化的解释。
- 以下是常见的 AI 相关术语词汇对应表（English -> 中文）：
  * Transformer -> Transformer
  * Token -> Token
  * LLM/Large Language Model -> 大语言模型
  * Zero-shot -> 零样本
  * Few-shot -> 少样本
  * AI Agent -> AI 智能体
  * AGI -> 通用人工智能
- 输入的需要翻译内容格式如下：

## {Title}
[{Title}]({Link})

{Abstract}

策略：

分三步进行翻译工作，并打印每步的结果：
1. 根据英文内容直译，保持原有格式，不要遗漏任何信息
2. 根据第一步直译的结果，指出其中存在的具体问题，要准确描述，不宜笼统的表示，也不需要增加原文不存在的内容或格式，包括不仅限于：
  - 不符合中文表达习惯，明确指出不符合的地方
  - 语句不通顺，指出位置，不需要给出修改意见，意译时修复
3. 根据第一步直译的结果和第二步指出的问题，重新进行意译，保证内容的原意的基础上，使其更易于理解，更符合中文的表达习惯，同时保持原有的格式不变
4. 翻译后的内容必须保留原始 Markdown 格式，标题行的标题保持英文原文，不要翻译，链接行的标题需要翻译成中文。

返回格式如下，"{xxx}"表示占位符：

### 直译
{直译结果}

***

### 问题
{直译的具体问题列表}

***

### 意译
```
{意译结果}
```

现在请按照上面的要求从第一行开始翻译以下内容为简体中文：
```
"""


class PaperInfo:
    """论文信息数据类"""
    def __init__(self, title: str, link: str, likes: int, arxiv_id: str = "", abstract: str = ""):
        self.title = title
        self.link = link
        self.likes = likes
        self.arxiv_id = arxiv_id
        self.abstract = abstract


class HuggingFaceScraper:
    """HuggingFace论文列表抓取器"""
    
    def __init__(self, base_url: str = BASE_URL, likes_threshold: int = LIKES_THRESHOLD):
        self.base_url = base_url
        self.likes_threshold = likes_threshold
    
    def get_current_week(self) -> str:
        """获取当前周的标识，格式如：2026-W04"""
        today = datetime.today()
        year, week, _ = today.isocalendar()
        return f"{year}-W{week:02d}"
    
    def scrape_papers(self, week: str = None) -> List[PaperInfo]:
        """
        抓取指定周的论文列表
        :param week: 周标识，如 "2026-W04"，默认为当前周
        :return: 论文信息列表
        """
        if week is None:
            week = self.get_current_week()
        
        url = f"{self.base_url}/papers/week/{week}"
        print(f"正在抓取: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            html_content = response.text
        except Exception as e:
            print(f"抓取失败: {e}")
            return []
        
        soup = BeautifulSoup(html_content, "html.parser")
        articles = soup.find_all("article")
        
        paper_list = []
        for article in articles:
            try:
                title = article.find("h3").get_text(strip=True)
                link = article.find("a")["href"]
                
                # 查找点赞数
                leading_nones = article.find_all("div", class_="leading-none")
                likes_div = None
                for item in leading_nones:
                    if item.get("class") == ["leading-none"]:
                        likes_div = item
                        break
                
                if likes_div is None:
                    continue
                
                likes = int(likes_div.get_text(strip=True))
                
                # 过滤点赞数低于阈值的论文
                if likes < self.likes_threshold:
                    break
                
                full_link = self.base_url + link
                arxiv_id = link.split('/')[-1]
                
                print(f"找到论文: {title} (点赞: {likes})")
                paper_list.append(PaperInfo(title, full_link, likes, arxiv_id))
                
            except Exception as e:
                print(f"解析文章失败: {e}")
                continue
        
        return paper_list


class ArxivClient:
    """Arxiv客户端，用于获取论文详细信息"""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def get_paper_info(self, arxiv_id: str) -> Optional[Dict[str, str]]:
        """
        根据arxiv ID获取论文信息
        :param arxiv_id: Arxiv论文ID
        :return: 包含标题、摘要、链接的字典
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(self.client.results(search))
            
            return {
                "title": result.title,
                "abstract": result.summary.replace('\n', ' ').strip(),
                "link": f"https://arxiv.org/abs/{arxiv_id}"
            }
        except StopIteration:
            print(f"未找到论文: {arxiv_id}")
        except Exception as e:
            print(f"获取论文信息失败: {e}")
        
        return None


class Translator:
    """翻译器，使用大模型进行翻译"""
    
    def __init__(self, model_name: str = "deepseek"):
        config = LLM_CONFIG.get(model_name)
        if not config or not config.get("api_key"):
            raise ValueError(f"未找到模型 {model_name} 的配置或API密钥")
        
        self.model_name = config["model_name"]
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
    
    def translate(self, text: str) -> str:
        """
        翻译文本
        :param text: 待翻译的英文文本
        :return: 翻译后的中文文本
        """
        try:
            messages = [
                {"role": "system", "content": TRANSLATION_PROMPT},
                {"role": "user", "content": text}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            
            result = response.choices[0].message.content
            return self._extract_translation(result)
            
        except Exception as e:
            print(f"翻译失败: {e}")
            return text
    
    def _extract_translation(self, text: str) -> str:
        """提取意译部分"""
        pattern = r'### 意译\s*(```)?(.+?)(```)?(?=###|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(2).strip()
        else:
            print("警告: 未找到意译部分，返回原始结果")
            return text


class PaperTranslatorPipeline:
    """论文翻译流水线"""
    
    def __init__(self):
        self.scraper = HuggingFaceScraper()
        self.arxiv_client = ArxivClient()
        self.translator = Translator()
    
    def format_english_content(self, title: str, link: str, abstract: str) -> str:
        """格式化英文内容"""
        return f"""## {title}
[{title}]({link})

{abstract}"""
    
    def generate_title(self, week: str = None) -> str:
        """生成文章标题，格式：每周AI论文速递（260119-260123）"""
        if week is None:
            today = datetime.today()
            start_of_week = today - timedelta(days=today.weekday())
        else:
            # 从week字符串解析日期，格式如 "2026-W05"
            year, week_num = week.split('-W')
            year = int(year)
            week_num = int(week_num)
            # 计算该周的周一日期
            jan_4 = datetime(year, 1, 4)
            start_of_week = jan_4 - timedelta(days=jan_4.weekday()) + timedelta(weeks=week_num - 1)
        
        # 计算周一到周五的日期
        weekdays = [start_of_week + timedelta(days=i) for i in range(5)]
        start_date = weekdays[0].strftime('%y%m%d')
        end_date = weekdays[-1].strftime('%y%m%d')
        
        return f"# 每周AI论文速递（{start_date}-{end_date}）"
    
    def run(self, output_file: str = None, week: str = None):
        """
        执行完整的流程
        :param output_file: 输出文件路径，默认使用日期命名
        :param week: 指定周，默认使用全局配置 TARGET_WEEK
        """
        # 如果没有指定周，使用全局配置
        if week is None:
            week = TARGET_WEEK
        
        print("=" * 60)
        print("开始执行论文翻译流程")
        if week:
            print(f"目标周: {week}")
        else:
            print("目标周: 当前周")
        print("=" * 60)
        
        # 1. 抓取HuggingFace论文列表
        print("\n步骤 1: 抓取HuggingFace论文列表...")
        papers = self.scraper.scrape_papers(week)
        print(f"共找到 {len(papers)} 篇论文\n")
        
        if not papers:
            print("未找到符合条件的论文，退出")
            return
        
        # 2. 获取Arxiv详细信息
        print("步骤 2: 获取Arxiv论文详细信息...")
        paper_details = []
        for paper in papers:
            print(f"正在获取: {paper.arxiv_id}")
            info = self.arxiv_client.get_paper_info(paper.arxiv_id)
            if info:
                paper_details.append(info)
        
        print(f"成功获取 {len(paper_details)} 篇论文详情\n")
        
        # 3. 翻译并输出
        print("步骤 3: 翻译论文标题和摘要...")
        
        # 确定输出文件名
        if output_file is None:
            today = datetime.today()
            start_of_week = today - timedelta(days=today.weekday())
            weekdays = [start_of_week + timedelta(days=i) for i in range(5)]
            output_file = f"{weekdays[0].strftime('%Y%m%d')}-{weekdays[-1].strftime('%Y%m%d')}.md"
        
        translated_count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            # 写入标题
            title = self.generate_title(week)
            f.write(title + "\n\n")
            
            for i, detail in enumerate(paper_details, 1):
                print(f"\n正在翻译 [{i}/{len(paper_details)}]: {detail['title'][:50]}...")
                
                # 格式化英文内容
                en_content = self.format_english_content(
                    detail['title'],
                    detail['link'],
                    detail['abstract']
                )
                
                # 翻译
                zh_content = self.translator.translate(en_content)
                
                # 写入文件
                f.write(zh_content + "\n\n")
                translated_count += 1
        
        print("\n" + "=" * 60)
        print(f"翻译完成！共翻译 {translated_count} 篇论文")
        print(f"输出文件: {output_file}")
        print("=" * 60)


def main():
    """主函数"""
    try:
        pipeline = PaperTranslatorPipeline()
        pipeline.run()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
