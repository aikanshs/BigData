Q1. 
Instructions to run on Q.1

Q.1 needs 5 arguments to be passed. 4 arguments for twitter
Arguments to be passed: consumer key, secret key, access token, secret access token

argument 5: Topic name to be used in kafka

Run zookeeper, kafka server and create a topic.
Start a consumer to that will dump messages to standard output

Run Elasticsearch, kibana and logstash to visulaize data in real time.

To run the project, run sparkStreaming.scala class on IntelliJ or create an assembly fat jar file and run the jar file with the below command:

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.0--class ClassName(sparkStreaming) PathToJarFile -- 5 arguments

The topic used for tweet was : "Trump"

Graphical plots have been uploaded that varied for 3 hours. Sentiment of maximum number of tweets were negtaive, 2nd largest number of sentiment were neutral and least number of tweets were positive.



Q2.
Instructions to run on Q2 on AWS cluster

Application location:

s3://ap-assign3/assignment3_2.11-0.1.jar

Adding class name and required jar file

--class SocialNetwork
--jars s3://ap-assign3/graphframes-0.7.0-spark2.4-s_2.11.jar

Arguments to be passed:

s3://ap-assign3/twitter_combined.txt	
s3://ap-assign3/output

Result: 

1.  The 5 nodes with the highest outdegree and the count of the number of outgoing edges in each were found and written in the output.2. The top 5 nodes with the highest indegree and the count of the number of incoming edges in each were found and written in the output.3. PageRank for each of the nodes were calculated and output of the top 5 nodes with the highest PageRank were found and written in the output.4. The connected components algorithm was run on a minigraph and the top 5 components with the largest number of nodes was written in the output.5. The triangle counts algorithm was ran on each of the vertices and the top 5 vertices with the largest triangle count was written in the output.

Note: If you try to pass same 2nd argument it'll return an error because the output already exists in that folder


