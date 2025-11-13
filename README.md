<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Hyperlink Prediction on Hypergraphs of Text</h3>

  <p align="center">
    This repository contains datasets and code for the paper Hyperlink Prediction on Hypergraphs of Text. Alessia Antelmi, Tiziano Citro, Dario De Maio, Daniele De Vinco, Valerio Di Pasquale, Mirko Polato, Carmine Spagnuolo.
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Hypergraphs have recently emerged as powerful tools for modeling high-order relationships in complex real-world data. Nevertheless, existing hyperlink prediction methods primarily emphasize the structural connectivity encoded by these structures, often overlooking the rich semantic information associated with nodes and relations, which can often be expressed in the form of text.

In this work, we propose a novel framework for hyperlink prediction on Hypergraphs of Text (HoTs), where both nodes and hyperedges are enriched with textual attributes. Our model jointly leverages these semantic and structural signals by combining hypergraph convolutional operators with cross-attention mechanisms that iteratively refine node and hyperedge representations.

Experimental results demonstrate that integrating semantic information from nodes and hyperedges with structural properties consistently improves performance over baselines relying solely on topology, hence highlighting the effectiveness of contextual representations for hyperlink prediction and opening new directions for semantic-aware hypergraph learning.

### Built With

* [![Python][Python-shield]][Python-url]
* [![PyTorch][PyTorch-shield]][PyTorch-url]
* [![PyTorch Geometric][PyG-shield]][PyG-url]
* [![PyTorch Lightning][Lightning-shield]][Lightning-url]
* [![DHG][DHG-shield]][DHG-url]

<!-- GETTING STARTED -->
## Getting Started

Follow these steps to set up and run **HPHoT (Hyperlink Prediction on Hypergraphs of Text)** locally.

### 🧩 Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.10+**
* **Git**
* (Optional) **CUDA** toolkit (if using GPU)

You can check your Python version with:
```bash
python --version
```
If you don’t have pip or venv, install them using:

```bash
python -m ensurepip --upgrade
```

⚙️ Installation
Clone the repository

```bash
git clone https://github.com/hypernetwork-research-group/hphot.git
cd hphot
(Recommended) Create and activate a virtual environment
```

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```
Install dependencies

```bash
pip install -r requirements.txt
```

<!-- USAGE EXAMPLES -->
## Usage

You can run all the experiments, including baselines and ablation studies, using the provided script:

```bash
./run.sh
```
This will:

* Train and evaluate all baseline models
* Perform ablation studies on different hypergraph configurations
* Save the results and logs in the runs/ directory

<!-- ROADMAP
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/github_username/repo_name/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=github_username/repo_name" alt="contrib.rocks image" />
</a> -->



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- Tech badges -->
[Python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/

[PyTorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/

[PyG-shield]: https://img.shields.io/badge/PyTorch%20Geometric-0080FF?style=for-the-badge&logo=pyg&logoColor=white
[PyG-url]: https://pytorch-geometric.readthedocs.io/

[Lightning-shield]: https://img.shields.io/badge/PyTorch%20Lightning-792EE5?style=for-the-badge&logo=lightning&logoColor=white
[Lightning-url]: https://lightning.ai/pytorch-lightning

[DHG-shield]: https://img.shields.io/badge/DHG%20(IMoonLab)-00B2A9?style=for-the-badge&logo=python&logoColor=white
[DHG-url]: https://github.com/iMoonLab/DeepHypergraph