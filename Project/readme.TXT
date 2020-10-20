Q1. 
Instructions to run Q.1

It needs 6 arguments to be passed. 4 arguments for twitter to generate oAuth credentials
Arguments to be passed: consumer key, consumer secret key, access token, secret access token, twitter topic, kafka topic

Run zookeeper, kafka server and create a topic.
Start a consumer to that will dump messages to standard output

Run Elasticsearch, kibana and logstash to visulaize data in real time.

To run the project, run sparkStreaming.scala class on IntelliJ or create an assembly fat jar file and run the jar file with the below command:

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.0--class ClassName(sparkStreaming) PathToJarFile -- 6 arguments

Insights:

The topic used for twitter was : "trump"
Graphical plots have been uploaded that varied for 3 hours.
Sentiment was constant over the 3 hours time period with of maximum number of tweets being negtaive (Approx 80%), 
2nd most common sentiment was neutral which are slightly more than number of positive tweets. 
Overall the sentiment for trump is negative.



Q2.
Instructions to run on Q2 on AWS cluster

spark-submit options: Adding class name and graphframes jar file

--class SocialNetwork
--jars s3://bigdatasu20/Assignment3/graphframes-0.7.0-spark2.4-s_2.11.jar

Application location: Jar file location

s3://bigdatasu20/Assignment3/assignment3_2.11-1.0.jar

Arguments to be passed:

s3://bigdatasu20/Assignment3/twitter_combined.txt	
s3://bigdatasu20/Assignment3/outputQ2	

Result:

1. User ID 3359851 follows 3373 people which is the highest
2. User ID 40981798 is the most followed person in twitter with 8660 followers
3. User ID 115485051 has the highest page rank of 308.41803928466965 and there is a corelation with indegree 
4. Componet 42949673045 has 44 nodes which is the highest
5. Triangle count for user ID 40981798 is 33528579 which is highest and we have observed that there is a corelation with indegree

Note: If you try to pass same 2nd argument it'll retun an error because the output already exists in that folder


