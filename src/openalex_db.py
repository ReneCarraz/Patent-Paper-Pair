import duckdb
import json
import pyalex
from typing import List, Tuple, Optional
from pyalex import Authors, Works
import pandas as pd
from requests import JSONDecodeError
from tqdm import tqdm


class OpenAlexLocalDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.create_tables()
        pyalex.config.email = None  # Put your email address here
        self.insert_batch_size = 1000

    def get_connection(self):
        return duckdb.connect(self.db_path)

    def create_tables(self):
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    inventor_full_name VARCHAR PRIMARY KEY,
                    author_full_name VARCHAR,
                    author_id VARCHAR,
                    institution_ids VARCHAR,
                    institution_names VARCHAR,
                    raw_data JSON
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS authors_already_fetched (
                    author_id VARCHAR PRIMARY KEY
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS works (
                    work_id VARCHAR PRIMARY KEY,
                    doi VARCHAR,
                    title VARCHAR,
                    abstract VARCHAR,
                    publication_date DATE,
                    referenced_works VARCHAR,
                    related_works VARCHAR,
                    raw_data JSON
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS authorships (
                    work_id VARCHAR,
                    author_id VARCHAR,
                    author_position VARCHAR,
                    full_name VARCHAR,
                    institutions VARCHAR,
                    raw_data JSON,
                    PRIMARY KEY (work_id, author_id)
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candidate_name ON candidates (inventor_full_name)
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS work_dois (
                    work_id VARCHAR PRIMARY KEY,
                    doi VARCHAR
                )
            """
            )

    def get_candidates(self, inventor_names: List[str]) -> pd.DataFrame:
        with self.get_connection() as conn:
            input_names = pd.DataFrame({"inventor_full_name": inventor_names})
            missing_inventors = (
                conn.sql(
                    """
                SELECT i.inventor_full_name
                FROM input_names i
                LEFT JOIN candidates c ON c.inventor_full_name = i.inventor_full_name
                WHERE c.inventor_full_name IS NULL
                """
                )
                .fetchdf()["inventor_full_name"]
                .tolist()
            )

            if missing_inventors:
                print(f"Fetching {len(missing_inventors)} new candidates")
                self._fetch_candidates(missing_inventors)

            result = conn.sql(
                """
                SELECT c.inventor_full_name, c.author_full_name, c.author_id, c.institution_ids, c.institution_names
                FROM input_names i
                JOIN candidates c ON i.inventor_full_name = c.inventor_full_name
                """
            ).fetchdf()

            return result

    def _fetch_candidates(self, inventor_names: List[str]):
        candidates_batch = []

        for inventor_name in tqdm(inventor_names, desc="Fetching candidates"):
            candidates = (
                Authors().search_filter(display_name=inventor_name).get()
            )
            for candidate in candidates:
                if not candidate:
                    continue

                institution_ids = set()
                institution_names = set()

                for affiliation in candidate.get("affiliations", []):  # type: ignore
                    institution = affiliation.get("institution", {})
                    if "id" in institution:
                        institution_ids.add(institution["id"])
                    if "display_name" in institution:
                        institution_names.add(institution["display_name"])

                candidates_batch.append(
                    (
                        inventor_name,
                        candidate["display_name"],  # type: ignore
                        candidate["id"],  # type: ignore
                        json.dumps(list(institution_ids)),
                        json.dumps(list(institution_names)),
                        json.dumps(candidate),
                    )
                )

            if len(candidates_batch) >= self.insert_batch_size:
                self._insert_candidates_batch(candidates_batch)
                candidates_batch = []

        if candidates_batch:
            self._insert_candidates_batch(candidates_batch)

    def _insert_candidates_batch(self, candidates_batch):
        with self.get_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO candidates
                    (inventor_full_name, author_full_name, author_id, institution_ids, institution_names, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    candidates_batch,
                )

                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                print(f"Error inserting batch: {e}")

    def get_works_by_author_ids(
        self,
        author_ids: List[str],
        limit_author_position: Optional[List[str]] = None,
        work_min_date: Optional[str] = None,
        work_max_date: Optional[str] = None,
        fetch: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with self.get_connection() as conn:
            input_authors = pd.DataFrame({"author_id": author_ids})
            if fetch:
                missing_author_ids = conn.sql(
                    """
                    SELECT ia.author_id
                    FROM input_authors ia
                    WHERE ia.author_id NOT IN (SELECT author_id FROM authors_already_fetched)
                """
                ).fetchdf()
                missing_author_ids = missing_author_ids["author_id"].tolist()

                if missing_author_ids:
                    print(f"Fetching data for {len(missing_author_ids)} new authors")
                    self._fetch_works_by_author_ids(
                        missing_author_ids,
                        work_min_date,
                        work_max_date,
                    )

            # get all authorships where the input_author is an author (limit to author_position if specified)
            # this will be used to get the relevant works
            get_input_authorships = conn.sql(
                """
                SELECT DISTINCT a.work_id, a.author_id, a.author_position
                FROM authorships a
                WHERE a.author_id IN (SELECT DISTINCT author_id FROM input_authors)
                """
            )

            # limit to author_position if specified
            if limit_author_position:
                get_input_authorships = get_input_authorships.filter(
                    duckdb.ColumnExpression("author_position").isin(
                        *[
                            duckdb.ConstantExpression(pos)
                            for pos in limit_author_position
                        ]
                    )
                )

            # get all works where the input_author is an author
            get_works = conn.sql(
                """
                SELECT DISTINCT w.work_id, w.doi, w.title, w.abstract, w.publication_date, w.referenced_works, w.related_works
                FROM works w
                WHERE w.work_id IN (SELECT DISTINCT ga.work_id FROM get_input_authorships ga)
                AND w.title IS NOT NULL
                AND w.abstract IS NOT NULL
                AND w.publication_date IS NOT NULL
                """
            )

            if work_min_date:
                get_works = get_works.filter(
                    duckdb.ColumnExpression("publication_date")
                    >= duckdb.ConstantExpression(work_min_date)
                )

            if work_max_date:
                get_works = get_works.filter(
                    duckdb.ColumnExpression("publication_date")
                    <= duckdb.ConstantExpression(work_max_date)
                )

            # get all the authorships for the selected works (regardless of whether they are in the input,
            # and regardless of author_position)
            get_authorships = conn.sql(
                """
                SELECT DISTINCT a.work_id, a.author_id, a.author_position, a.full_name, a.institutions
                FROM authorships a
                WHERE a.work_id IN (SELECT DISTINCT gw.work_id FROM get_works gw)
                """
            )

            works_df = get_works.fetchdf()
            authorships_df = get_authorships.fetchdf()

            return works_df, authorships_df

    def _process_work(self, work):
        return_work = (
            work["id"],
            work.get("doi"),
            work.get("title"),
            self.reconstruct_abstract(work.get("abstract_inverted_index")),
            work.get("publication_date"),
            json.dumps(work.get("referenced_works", [])),
            json.dumps(work.get("related_works", [])),
            json.dumps(work),
        )

        return_authorships = [
            (
                work["id"],  # type: ignore
                authorship["author"].get("id"),
                authorship.get("author_position"),
                authorship.get("author").get("display_name"),
                json.dumps(
                    [inst.get("id") for inst in authorship.get("institutions", [])]
                ),
                json.dumps(authorship),
            )
            for authorship in work.get("authorships", [])  # type: ignore
        ]

        return return_work, return_authorships

    def _fetch_works_by_author_ids(
        self,
        author_ids: List[str],
        fetch_work_min_date: Optional[str] = None,
        fetch_work_max_date: Optional[str] = None,
    ):
        filter_kwargs = {}
        if fetch_work_min_date:
            filter_kwargs["from_publication_date"] = fetch_work_min_date
        if fetch_work_max_date:
            filter_kwargs["to_publication_date"] = fetch_work_max_date

        works_batch = []
        authorships_batch = []
        author_ids_batch = []

        for author_id in tqdm(author_ids, desc="Fetching works"):
            try:
                works = []
                work_pager = (
                    Works()
                    .filter(author={"id": author_id.split("/")[-1]}, **filter_kwargs)
                    .paginate(per_page=200)
                )
                for work_page in work_pager:
                    works.extend(work_page)
            except JSONDecodeError:
                continue

            for work in works:
                new_work, new_authorships = self._process_work(work)
                works_batch.append(new_work)
                authorships_batch.extend(new_authorships)

            author_ids_batch.append(author_id)

            if len(works_batch) >= self.insert_batch_size:
                self._insert_works_batch(
                    works_batch, authorships_batch, author_ids_batch
                )
                works_batch = []
                authorships_batch = []
                author_ids_batch = []

        if works_batch:
            self._insert_works_batch(works_batch, authorships_batch, author_ids_batch)

    def _insert_works_batch(
        self, works_batch, authorships_batch, author_ids_batch=None
    ):
        with self.get_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.executemany(
                    """
                    INSERT INTO works
                    (work_id, doi, title, abstract, publication_date, referenced_works, related_works, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    works_batch,
                )

                conn.executemany(
                    """
                    INSERT INTO authorships
                    (work_id, author_id, author_position, full_name, institutions, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    authorships_batch,
                )

                if author_ids_batch:
                    conn.executemany(
                        """
                        INSERT INTO authors_already_fetched
                        (author_id)
                        VALUES (?)
                    """,
                        author_ids_batch,
                    )

                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                raise e

    def get_work_raw_data(self, work_ids: List[str]) -> pd.DataFrame:
        with self.get_connection() as conn:
            placeholders = ", ".join(["?"] * len(work_ids))
            query = f"""
                SELECT w.raw_data FROM works w WHERE w.work_id IN ({placeholders})
            """
            return conn.execute(query, work_ids).fetchdf()

    def get_works_by_work_ids(
        self, work_ids: List[str], fetch: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with self.get_connection() as conn:
            input_works = pd.DataFrame({"work_id": work_ids})
            missing_work_ids = conn.sql(
                """
                SELECT iw.work_id
                FROM input_works iw
                WHERE iw.work_id NOT IN (SELECT DISTINCT work_id FROM works)
            """
            ).fetchdf()

            missing_work_ids = missing_work_ids["work_id"].tolist()

            if missing_work_ids and fetch:
                print(f"Found {len(missing_work_ids)} missing works")
                self._fetch_works_by_work_ids(missing_work_ids)

            works_df = conn.execute(
                """
                SELECT w.work_id, w.doi, w.title, w.abstract, w.publication_date, w.referenced_works, w.related_works
                FROM input_works iw
                JOIN works w ON iw.work_id = w.work_id
            """
            ).fetchdf()

            authorships_df = conn.execute(
                """
                SELECT a.work_id, a.author_id, a.author_position, a.full_name, a.institutions
                FROM input_works iw
                JOIN authorships a ON iw.work_id = a.work_id
            """
            ).fetchdf()

            return works_df, authorships_df

    def _fetch_works_by_work_ids(self, work_ids: List[str]):
        works_batch = []
        authorships_batch = []

        fetch_batch_size = 50
        with tqdm(total=len(work_ids), desc="Fetching works") as pbar:
            for i in range(0, len(work_ids), fetch_batch_size):
                work_id_batch = work_ids[i : i + fetch_batch_size]
                fetched_works = (
                    Works()
                    .filter(
                        ids={
                            "openalex": "|".join(
                                [id.split("/")[-1] for id in work_id_batch]
                            )
                        }
                    )
                    .get(per_page=50)
                )

                works = {}
                for fetched_work in fetched_works:
                    works[fetched_work["id"]] = fetched_work  # type: ignore

                for work_id in work_id_batch:
                    work = works.get(work_id, {"id": work_id})
                    new_work, new_authorships = self._process_work(work)
                    works_batch.append(new_work)
                    authorships_batch.extend(new_authorships)

                pbar.update(fetch_batch_size)

                if len(works_batch) >= self.insert_batch_size:
                    self._insert_works_batch(works_batch, authorships_batch)
                    works_batch = []
                    authorships_batch = []

        if works_batch:
            self._insert_works_batch(works_batch, authorships_batch)

    def reconstruct_abstract(self, inverted_index: dict):
        if not inverted_index:
            return None

        word_positions = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions[pos] = word

        max_position = max(word_positions.keys())
        abstract = []
        for i in range(max_position + 1):
            if i in word_positions:
                abstract.append(word_positions[i])

        return " ".join(abstract)

    def get_work_dois(self, work_ids: List[str]) -> pd.DataFrame:
        with self.get_connection() as conn:
            input_works = pd.DataFrame({"work_id": work_ids})
            missing_work_ids = conn.sql(
                """
                SELECT iw.work_id
                FROM input_works iw
                LEFT JOIN work_dois wd ON iw.work_id = wd.work_id
                LEFT JOIN works w ON iw.work_id = w.work_id
                WHERE wd.work_id IS NULL AND w.work_id IS NULL;
            """
            ).fetchdf()

            missing_work_ids = missing_work_ids["work_id"].tolist()

            if missing_work_ids:
                print(f"Fetching DOIs for {len(missing_work_ids)} missing works")
                self._fetch_work_dois(missing_work_ids)

            # Fetch work DOIs
            work_dois_df = conn.execute(
                """
                SELECT iw.work_id, COALESCE(w.doi, wd.doi) as doi
                FROM input_works iw
                LEFT JOIN works w ON iw.work_id = w.work_id
                LEFT JOIN work_dois wd ON iw.work_id = wd.work_id
                WHERE w.doi IS NOT NULL OR wd.doi IS NOT NULL;
            """
            ).fetchdf()

            return work_dois_df

    def _fetch_work_dois(self, work_ids: List[str]):
        dois_batch = []

        fetch_batch_size = 100
        with tqdm(total=len(work_ids), desc="Fetching work DOIs") as pbar:
            for i in range(0, len(work_ids), fetch_batch_size):
                work_id_batch = work_ids[i : i + fetch_batch_size]
                fetched_works = (
                    Works()
                    .filter(
                        ids={
                            "openalex": "|".join(
                                [id.split("/")[-1] for id in work_id_batch]
                            )
                        }
                    )
                    .select(["id", "doi"])
                    .get(per_page=fetch_batch_size)
                )

                works = {}
                for fetched_work in fetched_works:
                    works[fetched_work["id"]] = fetched_work  # type: ignore

                for work_id in work_id_batch:
                    work = works.get(work_id, {"id": work_id})
                    dois_batch.append((work["id"], work.get("doi")))

                pbar.update(fetch_batch_size)

                if len(dois_batch) >= self.insert_batch_size:
                    self._insert_work_dois_batch(dois_batch)
                    dois_batch = []

        if dois_batch:
            self._insert_work_dois_batch(dois_batch)

    def _insert_work_dois_batch(self, dois_batch):
        with self.get_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO work_dois
                    (work_id, doi)
                    VALUES (?, ?)
                """,
                    dois_batch,
                )

                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                raise e
