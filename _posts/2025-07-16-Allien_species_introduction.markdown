---
layout: post
read_time: true
show_date: true
title:  Analysing allien species introduction in the Balearic Islands
date:   2021-03-12 13:32:20 -0600
description: Some neural network optimization algorithms mostly to implement momentum when doing back propagation.
img: posts/Allien_species/cover.jpeg
tags: [R, allien species, Time series, GLMs, unsupervised learning, hierarchical clustering]
author: Iván Cortés, Marcello Cerrato
github: Ivancf1995/Historic_Alien_Flora
mathjax: yes # leave empty or erase to prevent the mathjax javascript from loading
toc: yes # leave empty or erase for no TOC
---

Extensive material and methods and results from this study can be read in the [original manuscript](https://doi.org/10.1007/s10531-023-02620-z) in the Journal [Biodiversity and Conservation](https://link.springer.com/journal/10531)

In this post we will focus on the introduction of plant allien species in the Balearic Islands, and how we can use machine learning to analyse future trends based on historical data. 

The main challenge in this study was to gather all publicly available data regarding introductions and think about the visualizations that would be most useful to understand the trends in the introduction of the species. From the data we gathered (one row per citation) we characterized information about the taxa (life form, taxonomic family, biogeographical origin) and about the introduction (path, date, habitat). We used this information to carry out **Generalised Linear Models** as follows:

```r
Habitat_predicted_data=expand.grid(
  Fecha=c(1800:2100),
  Habitat=Habitat_datamodel$Habitat)

Habitat_year_model=glm(N~Fecha+Habitat,
                     family=poisson(link="log"),
                     data=Habitat_datamodel)
Habitat_year_model_int=glm(N~Fecha*Habitat,
                         family=poisson(link="log"),
                         data=Habitat_datamodel)

AIC(Habitat_year_model_int,
    Habitat_year_model)

summary(aov(Habitat_year_model_int))


pscl::pR2(Habitat_year_model_int)[4]#0.81

car::Anova(Habitat_year_model_int,type="III",singular.ok =TRUE)
```

For each variable, we used Poisson family as indicated for count data, testing model signicance with a Chi-squared test. The results showed that the model was significant, and we could use it to predict future trends in the introduction of allien species. We used car::Anova to get the significance of each variable in the model, and pscl::pR2 to get the pseudo R-squared value of the model.

Defined the most accurate model, we used it to make it out predictions:

```r
predicted_intro=predict(Habitat_year_model_int,
                        newdata=Habitat_predicted_data,
                        type="response",
                        se.fit = TRUE)
```

Note se.fit to TRUE to get the standard error of the predictions, which we used to calculate confidence intervals for the predictions. Plotting was carried out as follows:

```r
Habitat_year_plot=Habitat_plot_data%>%
  ggplot(aes(x=Fecha,y=fit,col=Habitat))+
  geom_vline(aes(xintercept=2023),lty=2,col="#a9a9a9")+
  geom_line(lty=2)+
  geom_ribbon(aes(ymin=fit-se.fit,ymax=fit+se.fit,
                  fill=Habitat,
  ),alpha=0.2,col=NA)+
  geom_point(data=Habitat_datamodel,alpha=0.2,
            aes(y=N))+
  ylab("Number of taxa")+
  labs(title="Habitat",
       subtitle=paste0("Factor sig. (Df= 6, F= 239.53, p< 0.001); Interaction sig. (Df=6 , LR= 58.64, p< 0.001)"),
       caption=paste0("Model p-value< 0.001; R2: 0.88"))+
  coord_cartesian(xlim=c(1800,2150))+
  scale_colour_manual(name="Blackburn index",
                      values = fun_color_range(length(levels(as.factor(Habitat_datamodel$Habitat)))))+
  scale_fill_manual(name="Blackburn index",
                    values = fun_color_range(length(levels(as.factor(Habitat_datamodel$Habitat)))))+
  theme_minimal()+
  theme(legend.position = "top",
        plot.title=element_text(hjust=0.5),
        plot.subtitle=element_text(size=8, hjust=0.5, face="italic"),
        text=element_text(family="Helvetica"))

Habitat_year_label_plot=directlabels::direct.label.ggplot(Habitat_year_plot,
                                                          list("last.points",
                                                               cex=0.5,
                                                               fontface="bold"))
```

Although model statistics could be automatically extracted, we decided to include them in the plot as a subtitle for clarity.  Note the inclusion of se.fit in the geom ribbon to include confidence intervals in the plot. On the other hand, we chosed to use directlabels library to label the lines in the plot, which allows for a cleaner visualization without overlapping labels. The final plot is shown below:

![Allien species introduction in the Balearic Islands](./assets/img/posts/Allien_species/trends.jpg)

We can observe that the introduction of allien species in the Balearic Islands has been increasing over time. The model predicts that this trend will continue in the future, with a significant increase in the number of introductions in the coming years, which could reach more than 1000 species, representing >50% of the total number of plant species in the Balearic Islands.

<tweet>If current trends in allien species introductions continue, by 2100 exotic species could represent >50% of the total number of plant species in the Balearic Islands.</tweet>

On the other hand, we can see that habitats related to human disturbance are more prone to introuctions, as well as some origins like Capense and neotropical among Cactaceae and Asteraceae family. The main introduction path is ornamental, no unintencional, so conservation efforts should focus on controlling them regualting their importation. 

<tweet>In the Balearic Islands main introduction pathways of plant allien speies are ornamental, no unintentional, highlighting the need for stricter regulations on plant imports.</tweet>

We also aimed to analyse if allien species introductions were similar among islands. To do so, wu used unspervised machine learnign with hierarchical clustering to group islands based on the presence of allien species. We transformed islands to columns and taxa to rows, indicating 1/0 for presence/absence of the species in each island, and then calulated the distance matrix using Euclidean distance. The clustering was done using Ward's method, which is a hierarchical clustering method that minimizes the variance within clusters.

<p style="text-align:center">
\(\Delta(A, B) = \dfrac{|A| \cdot |B|}{|A| + |B|} \cdot \left\| \bar{x}_A - \bar{x}_B \right\|^2\)
</p>

<p>
where A and B are the clusters, \(|A|\) and \(|B|\) are the number of elements in each cluster, and \(\bar{x}_A\) and \(\bar{x}_B\) are the centroids of each cluster. The distance between clusters is calculated as the product of the number of elements in each cluster divided by the sum of the number of elements in both clusters, multiplied by the squared distance between the centroids of each cluster. 
</p>

```r
A_PCA=A_islas[2:7]

A_PCA[is.na(A_PCA)]=0

Data_matrix=t(A_PCA)

colnames(Data_matrix)=A_islas[1]$Taxa

d=dist(Data_matrix,
       method="euclidean")

clustering=hclust(d, method="ward.D")

```

We then plotted the dendrogram to visualize the clustering of islands based on the presence of allien species:

```r
ddata_x <- dendro_data(as.dendrogram(clustering))
labs <- label(ddata_x)
p2 <- ggplot(segment(ddata_x)) +
  geom_segment(aes(x=x, y=y, xend=xend, yend=yend))+
  geom_text(data=label(ddata_x),
            aes(label=label, x=x, y=0),colour=NA)+
  ylab("Euclidean distance")+
  xlab("")+
  coord_flip()+
  theme_blank()+
  theme(text=element_text(family="Helvetica"),
        axis.text.y = element_blank())



p2
```

The resulting plot shows that Ibiza and formentera are the most similar islands, sharing a common set of allien species. Mallorca and Menorca are also similar, btu different from minor islands, suggestign that island size plays a major role in the introduction, probably to major demographic pressure and human disturbance. 

![Allien species introduction in the Balearic Islands by island](./assets/img/posts/Allien_species/dendro.jpg)


And that's all for this post! We hope you found it interesting and informative. If you have any questions or comments, feel free to reach out to us.
