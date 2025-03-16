package model

// TweetRecord represents a tweet with user ID, step, and opinion
type TweetRecord struct {
	UserID  int
	Step    int
	Opinion float64
}

// Tweet represents a tweet object
type Tweet struct {
	User    int
	Step    int
	Opinion float64
}

// ToRecord converts a Tweet to a TweetRecord
func (t *Tweet) ToRecord() TweetRecord {
	return TweetRecord{
		UserID:  t.User,
		Step:    t.Step,
		Opinion: t.Opinion,
	}
}
