\let\Begin\begin
\let\End\end
\newenvironment{widefig}{\renewenvironment{figure}{\begin{figure*}[!tb]\centering}{\end{figure*}}}{}

\newcommand{\e}{\epsilon}
\newcommand{\X}{\mathcal{X}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}

\newcommand{\argmin}{\mathop{\mathrm{arg\,min}}}
\newcommand{\argmax}{\mathop{\mathrm{arg\,max}}}

\newcommand{\suml}{\sum\limits}
\newcommand{\sumnl}{\sum\nolimits}

\newcommand{\abs}[1]{\left|{#1}\right|}
\newcommand{\norm}[1]{\left\|{#1}\right\|}
\newcommand{\floor}[1]{\left\lfloor{#1}\right\rfloor}
\newcommand{\ceil}[1]{\left\lceil{#1}\right\rceil}
\newcommand{\I}[1]{\mathbb{I}(#1)}
\newcommand{\set}[1]{\left\{#1\right\}}

\renewcommand{\c}[1]{\overline{#1}}
\renewcommand{\v}[1]{\mathbf{\mathrm{#1}}}
\newcommand{\shat}[1]{\vphantom{#1}\smash[t]{\hat{#1}}}
\newcommand{\ts}[1]{^\mathrm{#1}}

\newcommand{\der}[2]{\frac{\mathrm{d}{#1}}{\mathrm{d}{#2}}}
\newcommand{\pder}[2]{\frac{\partial{#1}}{\partial{#2}}}

\renewcommand{\P}[2]{\mathbb{P}_{#1}\left[{#2}\right]}
\newcommand{\E}[2]{\mathbb{E}_{#1}\left[{#2}\right]}
\newcommand{\V}[2]{\mathbb{V}_{#1}\left[{#2}\right]}
