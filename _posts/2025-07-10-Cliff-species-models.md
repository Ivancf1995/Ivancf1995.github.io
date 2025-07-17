---
layout: post
read_time: true
show_date: true
title: "Approaching cliff species distribution using RandomForest "
date: 2025-06-16
img: posts/Species_distribution_cliffs/post-cliffs.png
tags: [R, species distribution modelling, machine learning, artificial intelligence, consrvation]
category: manuscript
github: jrb772/Cliff_BalearicIslands_ENM
author: Iván Cortés Fernández
description: "Predicting "
---

Species Distribution modelling (SDM) is a powerful tool that can be used to understand the distribution of certain species in a given area and be used to inform conservation efforts, especially for species that are threatened or endangered. In this post, we will discuss how we used Random Forests (RF) to model the distribution of cliff species in the Balearic Islands, Spain. specific information about methods and results can be found in the [original manuscript](https://doi.org/10.1016/j.baae.2024.08.001) in the Journal [Basic and Applied Ecology](https://www.sciencedirect.com/journal/basic-and-applied-ecology).

SDMs are supervised learning algortihms that require ocurrence data as response variable and environmental variables as predictors. In this case, the response variable was the presence of cliff species, and the predictors were environmental variables such as altitude, slope, aspect and biolcimatic data. We used [WorldClim](https://www.worldclim.org/) data to obtain the environmental variables. 

Data for the study was obtained from the [Global Biodiversity Information Facility (GBIF)](https://www.gbif.org/), which provides access to a wide range of biodiversity data. The data was filtered to include only cliff species in Mallorca, and the coordinates were used to create a spatial dataset. This dataset was enlarged with own data collected in the field. Maps were created using [QGIS](https://www.qgis.org/en/site/). At the end, 20 species were selected for the analysis, representing over ~3000 occurences. 

![GBIF records of selected cliff species](./assets/img/posts/Species_distribution_cliffs/GBIF_records.jpg)

A challenge and one of the most sensible parts of Species distribution modelling is the creation of absence data, which is necessary to train the model. In this case, we generated buffers of 
1km areound the presence points and used these buffers to create pseudo-absence points. This method is widely used in SDMs, but it is important to note that it can introduce bias in the model if not done correctly. We used a random sampling approach to select the pseudo-absence points, ensuring that they were not too close to the presence points.

```r
library(terra)
# Occurrences and Raster loading ####
Species_data <- vect("~/Cliff_Occurrences_ESTR25831.shp")
Predictive_rasters <- rast("~/Predictive_Europe.tif")
Europe_Limits <- vect("~/Coastline_Europe_25381.shp")
Balearic_Limits <- vect("~/Coastline_BalearicIslands_25831.shp")
# Pseudo-Absence calculation ####
absence_multiplier = 2
buffers <- buffer(Species_data, 1000)

Background_area <- erase(Limites_Europe, aggregate(buffers))

Absence_points <- terra::spatSample(Background_area, 
                                    size = absence_multiplier*nrow(Species_data),
                                    method = "random")


```
Using this code, we cropped the buffers from the background area (Europe limits) and sampled random points from it. 

Then, we extracted the environmental variables from the presence and pseudo-absence points using the `extract` function from the `terra` package. This allowed us to create a dataset with the presence and absence points, along with the environmental variables.

```r
extracted_predictors_raw <- as.data.frame(terra::extract(Predictive_rasters, occs, xy=T, bind=T))

extracted_predictors_raw2 <- extracted_predictors_raw %>% 
  select(-"Presence",-"x",-"y", -"Geo", -"Land_Use")
extracted_predictors <- extracted_predictors_raw %>% 
  select("Presence","x","y","Geo","Land_Use")

# Fill missing data
extracted_NA_kNN <- caret::preProcess(extracted_predictors_raw2,
                                      method="bagImpute")
missing_points <- predict(extracted_NA_kNN, extracted_predictors_raw2)

cliff_data <- cbind(extracted_predictors, missing_points)
```

Note the use of `caret::preProcess` to fill missing data using bagInputing (based on Regression Trees), but other options are available (mean, median, knn or linear imputation ). This is a common approach to handle missing data in SDMs, as it allows us to use all available data without losing information, as some supervised learning algorithms do not handle missing data well.

Although some altogrithm can handle well with correlated variables, it is a good practice to check for multicollinearity in the predictors. In this case, we used the `vif` function from the `usdm` package to calculate the Variance Inflation Factor (VIF) and remove highly correlated variables.

```r
loop_val <- 100
while (loop_val >= 10) {
  Numeric_spg <- cliff_data %>%
    select(-"Presence",-"x",-"y") %>%
    select_if(.,is.numeric)
  
  VIF_numeric <- usdm::vif(Numeric_spg[,6:length(Numeric_spg)]) %>%
    arrange(VIF)
  VIF_numeric 
  
  Selected_variables <- VIF_numeric[-c(length(VIF_numeric$VIF)),]
  
  cliff_selected_data <- cliff_data %>%
                  select("Presence","x","y",names(Numeric_spg[,1:5]),
                         Selected_variables$Variables,names(Not_Numeric_spg))
  
  loop_val <- VIF_numeric[c(length(VIF_numeric$VIF)),2]
  print(paste("max VIF at", loop_val))
}
```

Finally, we used the `ranger` package to train a Random Forest model on the data. The model was trained using 40-fold cross-validation and repeated 40 times to ensure robustness. The model was then used to predict the distribution of cliff species in Mallorca. We opted to use RandomForest as it is a flexible algorithm that can handle non-linear relationships, less restrictive assumptions about the data than Generalised Linear Models, and is robust to overfitting.

```r
# Random Forest model ####
cliff_selected_data$Presence <- as.factor(cliff_selected_data$Presence)
levels(cliff_selected_data$Presence) <- c("Ausence","Presence")

rf.task <-  makeClassifTask(data = cliff_selected_data, target = "Presence")
res <- tuneRanger(rf.task, measure = list(multiclass.brier), num.trees = 500,
                  num.threads = 20, iters = 100, save.file.path = NULL)

cliff_RF_model <- caret::train(
  cliff_selected_data[,-1],     # This way the factor rasters are keep together
  as.factor(cliff_selected_data$Presence),
  method = "ranger",
  trControl = trainControl(method = "repeatedcv",   # k-fold cross-validation
                           number = 40,             # 40 folds
                           repeats = 40,            # repeated 40 times
                           allowParallel = T,
                           classProbs=T,
                           returnData = T,
                           savePredictions = "final"),
  tuneGrid = expand.grid(mtry = res$recommended.pars[,1],
                         min.node.size = res$recommended.pars[,2],
                         splitrule = "gini"),
  num.trees = 500,
  num.threads = 20,
  importance = 'impurity'
)
```

After training the model, we evaluated its performance using the `confusionMatrix` function from the `caret` package. This allowed us to assess the accuracy of the model and its ability to correctly classify presence and absence points.

```r
conf_RF <- confusionMatrix(cliff_RF_model)$table
```

One of the most beatifoul things about randomForest is that it provides a measure of variable importance, which can be used to understand which environmental variables are most important for the distribution of cliff species. We used `importance` function from the `ranger` package to obtain the variable importance scores and plotted them using `ggplot2`.

```r
importance_scores <- as.data.frame(importance(cliff_RF_model$finalModel))
importance_scores$Variable <- rownames(importance_scores)
importance_scores <- importance_scores %>%
  arrange(desc(IncNodePurity))
```

With a little bit of ggplot magic, we can visualize the importance of each variable in the model

![Variable importance plot](./assets/img/posts/Species_distribution_cliffs/SHAP.jpg)



Also, we are able to see how the model undestands the relationship between the environmental variables and the presence of cliff species. We can use the `partialPlot` function from the `ranger` package to visualize the partial dependence of each variable on the model predictions.

```r
vars <- c("BIO9", "BIO19", "Slope", "BIO4", "Elevation", "BIO3")

plots <- lapply(vars, function(var) {
  pd <- partial(cliff_RF_model, pred.var = var, train = my_data, prob = TRUE)
  autoplot(pd) + 
    ggtitle(var) +
    theme_minimal()
})

grid.arrange(grobs = plots, ncol = 3)
```


![Partial dependence plot](./assets/img/posts/Species_distribution_cliffs/PDP.jpg)

We can observe that the model predicts relationships between probability of presence and onvironmental variables, such as BIO9 (Mean Temperature of Driest Quarter), BIO19 (Precipitation of Coldest Quarter), Slope, BIO4 (Temperature Seasonality), Elevation and BIO3 (Isothermality) quite different than lineal or logistic, which is one of the advantages of using tree-based models.

Finally, we can use the trained model to predict the distribution of cliff species in Mallorca. We used the `predict` function from the `terra` package to make predictions on the environmental variables raster data, generating directly a raster file which can be easily visualized in QGIS or any other GIS software.

![Predicted distribution of cliff species in Mallorca](./assets/img/posts/Species_distribution_cliffs/current_map.jpg)

We can observe how the model predicts the distribution of cliff species. Mallorca, because of its high mountainous areas in comparison to the other islands, has a high probability of presence of cliff species in the north and northwest areas, where the highest cliffs are located. The model also predicts lower probabilities of presence in the southern and eastern areas, where the cliffs are less pronounced.

Finally, we can change environmental raster variables to predict how the distribution of cliff species would change under different scenarios of climate change.

![Predicted distribution of cliff species in Mallorca under climate change scenarios](./assets/img/posts/Species_distribution_cliffs/future_map.jpg)

In comparison with the current distribution, we can see that distribution area is considerably reduced in future scenarios of climate change, even in the most moderate (SSP1-2.6), as a result of modifications temperatures and precipitation patterns. This highlights the importance of using SDMs to predict the impact of climate change on species distribution and inform conservation efforts.

<tweet>Using Random Forests to model the distribution of cliff species in Mallorca, we predict a significant reduction in their distribution area under climate change scenarios, enhancing their conservation priority</tweet>

*Note: we have ommited train-test split and hyperparameter tuning for simplicity, but it is a good practice to avoid overfitting and achieve better generalization capabilities*