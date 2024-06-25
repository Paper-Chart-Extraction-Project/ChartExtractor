# ChartExtractor
ChartExtractor is a computer vision program that extracts data from smartphone images of paper medical charts.

## Why Does This Exist?

The availability of readily accessible, digital medical data offers two major benefits to institutions: it provides real-time access for informed medical decisions and retrospective data for research. This data allows researchers to:

1. Study statistical associations to uncover socioeconomic, genetic, and environmental factors of health.
2. Quantify how medicine is being used and how practice varies geographically.
3. Create retrospective studies that can then be used for prospective randomized control trials to study causality.

EMRs are now widespread in the west, with 96% adoption in American hospitals and 78% in office-based clinics [National Trends in Hospital and Physician Adoption of Electronic Health Records](https://www.healthit.gov/data/quickstats/national-trends-hospital-and-physician-adoption-electronic-health-records). Their adoption has allowed groups like MPOG and NACOR to form, which collate massive datasets of medical records. These datasets enable researchers to perform large meta-analyses on millions of data points, including studies of uncommon and poorly understood conditions by providing enough data for statistically significant analysis.

Additionally, a physician caring for a patient recovering in an ICU can easily access a hyperlinked version of their surgical case and history, speeding up the application of interventions or tests.

Although these benefits are exceptionally valuable, adopting electronic medical records (EMRs) has four distinct challenges [Challenges of Implementing Electronic Health Records in Resource-Limited Settings](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5654179/):

1. EMRs are costly.
2. EMRs require training and human capital to implement and maintain.
3. EMRs require sufficient electrical and internet infrastructure to function effectively.
4. EMRs necessitate significant workflow changes.

Due to these challenges, developing nations have been slow to adopt EMRs.

ChartExtractor, a project launched in 2019, offers a non-EMR solution to bridge the gap. This free, easily accessible smartphone application digitizes paper charts, requiring minimal changes to clinical operations. 

