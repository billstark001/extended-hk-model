package model

import (
	"os"
	"sync"

	"github.com/vmihailenco/msgpack/v5"
)

// EventLogger handles logging of events during simulation
type EventLogger struct {
	Filename  string
	BatchSize int
	queue     chan any
	stopFlag  chan struct{}
	wg        sync.WaitGroup
	lock      sync.Mutex
	file      *os.File
}

// NewEventLogger creates a new event logger
func NewEventLogger(filename string, batchSize int) (*EventLogger, error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, err
	}

	logger := &EventLogger{
		Filename:  filename,
		BatchSize: batchSize,
		queue:     make(chan any, batchSize*2),
		stopFlag:  make(chan struct{}),
		file:      file,
	}

	logger.wg.Add(1)
	go logger.worker()

	return logger, nil
}

// LogEvent adds an event to the queue for logging
func (l *EventLogger) LogEvent(event any) {
	select {
	case l.queue <- event:
		// Event queued successfully
	case <-l.stopFlag:
		// Logger is stopping, discard event
	}
}

// worker processes the event queue
func (l *EventLogger) worker() {
	defer l.wg.Done()

	batch := make([]any, 0, l.BatchSize)

	for {
		select {
		case event := <-l.queue:
			batch = append(batch, event)

			if len(batch) >= l.BatchSize {
				l.writeBatch(batch)
				batch = make([]any, 0, l.BatchSize)
			}

		case <-l.stopFlag:
			// Write any remaining events
			if len(batch) > 0 {
				l.writeBatch(batch)
			}
			return
		}
	}
}

// writeBatch writes a batch of events to the file
func (l *EventLogger) writeBatch(batch []any) {
	l.lock.Lock()
	defer l.lock.Unlock()

	for _, event := range batch {
		data, err := msgpack.Marshal(event)
		if err != nil {
			// Log error or handle it as appropriate
			continue
		}

		_, err = l.file.Write(data)
		if err != nil {
			// Log error or handle it as appropriate
		}
	}

	l.file.Sync()
}

// Stop closes the logger
func (l *EventLogger) Stop() {
	l.lock.Lock()

	// Signal worker to stop
	close(l.stopFlag)

	// Wait for worker to finish
	l.lock.Unlock()
	l.wg.Wait()

	// Close the file
	l.lock.Lock()
	defer l.lock.Unlock()
	l.file.Close()
}

// ReadMsgpackObjects reads msgpack objects from a file
func ReadMsgpackObjects(filename string) ([]any, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var objects []any
	decoder := msgpack.NewDecoder(file)

	for {
		var obj any
		err := decoder.Decode(&obj)
		if err != nil {
			break
		}
		objects = append(objects, obj)
	}

	return objects, nil
}
