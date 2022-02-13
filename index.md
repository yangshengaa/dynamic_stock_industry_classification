# Dynamic Stock Industry Classification

## Background

Stocks are classified into different sectors (Energy, Finance, Health Care, etc), and stocks within the same sectors are assumed to have similar behaviors (movement patterns and risk profiles). Fund managers worldwide demand a precise classification to control portfolio sector exposures and thus minimize risks brought by some specific sectors. This could be considered a sector-level diversification.  

The most widely used industry classifications are China CITIC (中信) and SWS Research (申万宏源) for China A-share. They provide a professional guideline for a long-term stock industry classification. However, the classification is fixed and fails to capture short-term correlations between stocks in different sectors, and thus fails to embody short-term co-movement risks between conventionally unrelated stocks. For example, company A in finance sector and company B in energy sector are considered uncorrelated, in a loose sense, conventionally. Due to a recent announcement of cooperation, their stock prices started to behave similarly. This particular risk could hardly be hedged against if the fund manager use a fixed industry classification scheme.  

Therefore, a dynamic industry classification is much more recommended for institutional traders, especially hedge fund portfolio managers on low-frequency trading strategies (change stock holdings each day, for instance).  

<center>
<figure>
<img src="out/graph_fig/cluster_demo_pmfg.png" alt="Trulli" style="width:70%">
<figcaption align = "center"><b>Trained zz1000 Member Stock Classification (2021/01/01 - 2021/12/31) </b></figcaption>
</figure>
</center>

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/yangshengaa/dynamic_stock_industry_classification/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

## Reference
