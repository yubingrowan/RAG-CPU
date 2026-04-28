"""
Kafka Index Consumer
Consumes indexing tasks from Kafka and processes them asynchronously
"""

import json
import asyncio
from kafka import KafkaConsumer
from indexing_pipeline import IndexingPipeline


class KafkaIndexConsumer:
    """Consumer for Kafka indexing tasks"""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "index_tasks",
        group_id: str = "index_consumer_group"
    ):
        """
        Initialize Kafka consumer
        
        Args:
            bootstrap_servers: Kafka server addresses
            topic: Topic to consume from
            group_id: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.indexing_pipeline = IndexingPipeline()
        self.consumer = None
    
    def _create_consumer(self):
        """Create Kafka consumer"""
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
    
    async def process_message(self, message):
        """
        Process a single indexing task message
        
        Args:
            message: Kafka message with indexing task
        """
        try:
            task_data = message.value
            print(f"Processing indexing task: {task_data.get('task_id', 'unknown')}")
            
            # Extract documents from task
            documents = task_data.get('documents', [])
            
            if not documents:
                print(f"No documents in task {task_data.get('task_id')}")
                return
            
            # Index documents
            result = await self.indexing_pipeline.index_documents(documents)
            
            print(f"Indexed {result['total_chunks']} chunks from {result['total_documents']} documents")
            
        except Exception as e:
            print(f"Error processing message: {e}")
    
    async def consume(self):
        """Consume and process indexing tasks from Kafka"""
        self._create_consumer()
        print(f"Started consuming from topic: {self.topic}")
        
        try:
            for message in self.consumer:
                await self.process_message(message)
                
        except KeyboardInterrupt:
            print("\nStopping consumer...")
        except Exception as e:
            print(f"Consumer error: {e}")
        finally:
            if self.consumer:
                self.consumer.close()
    
    def run(self):
        """Run the consumer (blocking)"""
        asyncio.run(self.consume())


if __name__ == "__main__":
    # Run the consumer
    consumer = KafkaIndexConsumer()
    consumer.run()
