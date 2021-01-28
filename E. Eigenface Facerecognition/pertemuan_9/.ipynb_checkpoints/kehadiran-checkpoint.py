import sqlite3

class Kehadiran():
    def __init__(self):
        self.conn = sqlite3.connect("attendance.db")
        self.conn.row_factory = self.dict_factory
        self.c = self.conn.cursor()
        
    def dict_factory(self, cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d
        
    def select_kehadiran(self, search=''):
        self.c.execute(
            """
            SELECT * FROM Kehadiran WHERE NamaKelas LIKE ?;
            """, ('%' + search + '%',))
        return self.c.fetchall()
    
    def create_kehadiran(self, record):
        try :
            self.c.execute(
            """
            INSERT INTO Kehadiran (
                              UserId,
                              JamMasuk,
                              NamaKelas,
                              JamKelasMulai,
                              JamKelasBerakhir,
                              Status,
                              Date)
                             VALUES (
                             ?, ?, ?, ?, ?, ?, ?
                             )
            """, (record['UserId'],
                 record['JamMasuk'],
                 record['NamaKelas'],
                 record['JamKelasMulai'],
                 record['JamKelasBerakhir'],
                 record['Status'],
                 record['Date']))
            self.conn.commit()
            return 'sucess'
        except Exception as e:
            return 'error :', e

    def delete_kehadiran(self, Id):
        try :
            self.c.execute(
            """
            DELETE FROM Kehadiran WHERE Id = ?
            """, (Id,))
            self.conn.commit()
            return 'success'
        except Exception as e:
            return 'error :', e
        
    def create_kehadiran_by_predictionid(self, record):
        # update data kehadiran if user exist

        self.c.execute(
        """
        SELECT * 
        FROM User 
        WHERE User.PredictionId = ?
        """, (record['PredictionId'],))

        user = self.c.fetchone() 

        if user is not None :
            try :
                self.c.execute("""
                INSERT INTO Kehadiran  (
                                  UserId,
                                  JamMasuk,
                                  NamaKelas,
                                  JamKelasMulai,
                                  JamKelasBerakhir,
                                  Status,
                                  Date
                              )
                              VALUES (
                                  ?, ?, ?, ?, ?, ?, ?
                              )

                """, (user['Id'], 
                      record['JamMasuk'], 
                      record['NamaKelas'], 
                      record['JamKelasMulai'], 
                      record['JamKelasBerakhir'], 
                      record['Status'], 
                      record['Date']))
                self.conn.commit()
                return 'success'
            except :
                return 'error'
        else :
            return 'user with PredictionId %d not found' % record['PredictionId']
        
        
    def update_kehadiran_selesai(self, NamaKelas, Date, JamKelasBerakhir):
        try :
            self.c.execute(
            """
            UPDATE Kehadiran SET 
                   JamKelasBerakhir = ?,
                   Status = 'selesai'
            WHERE NamaKelas = ? AND Date = ?
            """, (JamKelasBerakhir, NamaKelas, Date))
            self.conn.commit()
            return 'success'
        except Exception as e:
            return 'error :', e
        
        
    def select_nama_kelas(self):
        self.c.execute("""
        SELECT NamaKelas 
        FROM Kehadiran 
        GROUP BY NamaKelas 
        ORDER BY NamaKelas;
        """)
        return [item['NamaKelas'] for item in self.c.fetchall()]