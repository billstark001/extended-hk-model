package simulation

import (
	"database/sql"
	model "ehk-model/model"
	"fmt"

	_ "github.com/mattn/go-sqlite3"
	"github.com/vmihailenco/msgpack/v5"
)

type EventDB struct {
	db *sql.DB
}

// OpenEventDB 从文件打开数据库
func OpenEventDB(filename string) (*EventDB, error) {
	db, err := sql.Open("sqlite3", filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// 创建事件表
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS events (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			type TEXT NOT NULL,
			agent_id INTEGER NOT NULL,
			step INTEGER NOT NULL
		)
	`)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create events table: %w", err)
	}

	// 创建rewiring事件表
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS rewiring_events (
			event_id INTEGER PRIMARY KEY,
			unfollow INTEGER NOT NULL,
			follow INTEGER NOT NULL,
			FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
		)
	`)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create rewiring_events table: %w", err)
	}

	// 创建tweet事件表
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS tweet_events (
			event_id INTEGER PRIMARY KEY,
			agent_id INTEGER NOT NULL,
			step INTEGER NOT NULL,
			opinion REAL NOT NULL,
			is_retweet BOOLEAN NOT NULL,
			FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
		)
	`)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create tweet_events table: %w", err)
	}

	// 创建view_tweets事件表 (使用msgpack存储)
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS view_tweets_events (
			event_id INTEGER PRIMARY KEY,
			data BLOB NOT NULL,
			FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
		)
	`)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create view_tweets_events table: %w", err)
	}

	// 启用外键约束
	_, err = db.Exec("PRAGMA foreign_keys = ON")
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to enable foreign keys: %w", err)
	}

	return &EventDB{db: db}, nil
}

// Close 关闭数据库连接
func (edb *EventDB) Close() error {
	return edb.db.Close()
}

// StoreEvent 存储事件到数据库
func (edb *EventDB) StoreEvent(event *model.EventRecord) error {
	// 开始事务
	tx, err := edb.db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			tx.Rollback()
		}
	}()

	// 插入基本事件信息
	result, err := tx.Exec(
		"INSERT INTO events (type, agent_id, step) VALUES (?, ?, ?)",
		event.Type, event.AgentID, event.Step,
	)
	if err != nil {
		return fmt.Errorf("failed to insert event: %w", err)
	}

	// 获取自动生成的事件ID
	eventID, err := result.LastInsertId()
	if err != nil {
		return fmt.Errorf("failed to get last insert ID: %w", err)
	}

	// 根据事件类型处理具体的事件内容
	switch event.Type {
	case "Rewiring":
		if body, ok := event.Body.(model.RewiringEventBody); ok {
			_, err = tx.Exec(
				"INSERT INTO rewiring_events (event_id, unfollow, follow) VALUES (?, ?, ?)",
				eventID, body.Unfollow, body.Follow,
			)
			if err != nil {
				return fmt.Errorf("failed to insert rewiring event: %w", err)
			}
		} else {
			return fmt.Errorf("invalid RewiringEventBody type")
		}

	case "Tweet":
		if body, ok := event.Body.(model.TweetEventBody); ok {
			if body.Record == nil {
				return fmt.Errorf("tweet record is nil")
			}
			_, err = tx.Exec(
				"INSERT INTO tweet_events (event_id, agent_id, step, opinion, is_retweet) VALUES (?, ?, ?, ?, ?)",
				eventID, body.Record.AgentID, body.Record.Step, body.Record.Opinion, body.IsRetweet,
			)
			if err != nil {
				return fmt.Errorf("failed to insert tweet event: %w", err)
			}
		} else {
			return fmt.Errorf("invalid TweetEventBody type")
		}

	case "ViewTweets":
		if body, ok := event.Body.(model.ViewTweetsEventBody); ok {
			// 使用msgpack序列化ViewTweetsEventBody
			data, err := msgpack.Marshal(body)
			if err != nil {
				return fmt.Errorf("failed to marshal ViewTweetsEventBody: %w", err)
			}
			_, err = tx.Exec(
				"INSERT INTO view_tweets_events (event_id, data) VALUES (?, ?)",
				eventID, data,
			)
			if err != nil {
				return fmt.Errorf("failed to insert view tweets event: %w", err)
			}
		} else {
			return fmt.Errorf("invalid ViewTweetsEventBody type")
		}

	default:
		return fmt.Errorf("unknown event type: %s", event.Type)
	}

	// 提交事务
	return tx.Commit()
}

// DeleteEventsAfterStep 删除步骤大于等于指定值的所有事件
func (edb *EventDB) DeleteEventsAfterStep(step int) error {
	_, err := edb.db.Exec("DELETE FROM events WHERE step >= ?", step)
	if err != nil {
		return fmt.Errorf("failed to delete events: %w", err)
	}
	return nil
}

// GetEvents 获取所有事件（示例如何从数据库加载事件）
func (edb *EventDB) GetEvents() ([]*model.EventRecord, error) {
	rows, err := edb.db.Query(`
		SELECT e.id, e.type, e.agent_id, e.step FROM events e
		ORDER BY e.step ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to query events: %w", err)
	}
	defer rows.Close()

	var events []*model.EventRecord
	for rows.Next() {
		var id int64
		event := &model.EventRecord{}
		err := rows.Scan(&id, &event.Type, &event.AgentID, &event.Step)
		if err != nil {
			return nil, fmt.Errorf("failed to scan event: %w", err)
		}

		// 根据事件类型获取具体内容
		switch event.Type {
		case "Rewiring":
			var body model.RewiringEventBody
			err = edb.db.QueryRow(
				"SELECT unfollow, follow FROM rewiring_events WHERE event_id = ?", id,
			).Scan(&body.Unfollow, &body.Follow)
			if err != nil {
				return nil, fmt.Errorf("failed to scan rewiring event: %w", err)
			}
			event.Body = body

		case "Tweet":
			var body model.TweetEventBody
			var agentID, step int64
			var opinion float64
			var isRetweet bool

			err = edb.db.QueryRow(
				"SELECT agent_id, step, opinion, is_retweet FROM tweet_events WHERE event_id = ?", id,
			).Scan(&agentID, &step, &opinion, &isRetweet)
			if err != nil {
				return nil, fmt.Errorf("failed to scan tweet event: %w", err)
			}

			body.Record = &model.TweetRecord{
				AgentID: agentID,
				Step:    int(step),
				Opinion: opinion,
			}
			body.IsRetweet = isRetweet
			event.Body = body

		case "ViewTweets":
			var data []byte
			err = edb.db.QueryRow(
				"SELECT data FROM view_tweets_events WHERE event_id = ?", id,
			).Scan(&data)
			if err != nil {
				return nil, fmt.Errorf("failed to scan view tweets event: %w", err)
			}

			var body model.ViewTweetsEventBody
			err = msgpack.Unmarshal(data, &body)
			if err != nil {
				return nil, fmt.Errorf("failed to unmarshal view tweets event: %w", err)
			}
			event.Body = body
		}

		events = append(events, event)
	}

	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating events: %w", err)
	}

	return events, nil
}
