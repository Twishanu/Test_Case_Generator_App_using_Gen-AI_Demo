import os
import psycopg2
from dotenv import load_dotenv
from typing import List, Dict, Any
import json

load_dotenv()

class DBOperations:
    def __init__(self):
        self.db_params = {
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASS"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "database": os.getenv("DB_NAME")
        }
        self.connection = None
        self.connectDB()

    def connectDB(self):
        try:
            self.connection = psycopg2.connect(**self.db_params)
            print("Connection is successful...")
        except Exception as e:
            print(f"Error: {e} while connecting to the Postgres DB")

    def getChats(self) -> List[Dict[str, Any]]:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT chat_id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC")
                chats = cursor.fetchall()
                return [
                    {
                        "chat_id": row[0],
                        "title": row[1],
                        "created_at": row[2],
                        "updated_at": row[3]
                    }
                    for row in chats
                ]
        except Exception as e:
            print(f"Error: {e} fetching chats")
            return []
    
    def getChatMessages(self, chat_id: str) -> List[Dict[str, str]]:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT role, content, created_at FROM messages WHERE chat_id = %s ORDER BY created_at ASC",
                               (chat_id,))
                messages = cursor.fetchall()
                return [
                    {
                        "role": row[0],
                        "content": row[1],
                        "created_at": row[2]
                    }
                    for row in messages
                ]
        except Exception as e:
            print(f"Error {e} while fetching messages from this chat")
            return []

    def createChats(self, title: str = "Untitled chat") -> str:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("INSERT INTO chats (title) VALUES (%s) RETURNING chat_id",
                               (title,))
                chat_id = cursor.fetchone()[0]
                self.connection.commit()
                return chat_id
        except Exception as e:
            print(f"Error: {e} could not create a new chat")
            self.connection.rollback()
            return ""

    def update_chat_title(self, chat_id: str, new_title: str) -> bool:
        """Update the title of an existing chat"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE chats SET title = %s, updated_at = NOW() WHERE chat_id = %s",
                    (new_title, chat_id)
                )
                self.connection.commit()
                return True
        except Exception as e:
            print(f"Error updating chat title: {e}")
            self.connection.rollback()
            return False

    def addMessage(self, chat_id: str, role: str, content: str):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO messages (chat_id, role, content) VALUES (%s, %s, %s)",
                    (chat_id, role, content)
                )
                cursor.execute(
                    "UPDATE chats SET updated_at = NOW() WHERE chat_id = %s",
                    (chat_id,)
                )
                self.connection.commit()
        except Exception as e:
            print(f"Error adding message: {e}")
            self.connection.rollback()

    def delete_chat(self, chat_id: str) -> bool:
        """Delete chat and return success status for embedding cleanup"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM chats WHERE chat_id = %s", (chat_id,))
                self.connection.commit()
                return True
        except Exception as e:
            print(f"Error deleting chat: {e}")
            self.connection.rollback()
            return False

bot = DBOperations()