/*
Portions Copyright 2019-2021 ZomboDB, LLC.
Portions Copyright 2021-2022 Technology Concepts & Design, Inc. <support@tcdi.com>

All rights reserved.

Use of this source code is governed by the MIT license that can be found in the LICENSE file.
*/
//! Wrapper around Postgres' `pg_config` command-line tool
use eyre::{eyre, WrapErr};
use owo_colors::OwoColorize;
use serde_derive::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};
use std::process::Stdio;
use std::{
    collections::HashMap,
    io::ErrorKind,
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
};
use url::Url;

pub static BASE_POSTGRES_PORT_NO: u16 = 28800;
pub static BASE_POSTGRES_TESTING_PORT_NO: u16 = 32200;

// These methods were originally in `pgx-utils`, but in an effort to consolidate
// dependencies, the decision was made to package them into wherever made the
// most sense. In this case, it made the most sense to put them into this
// pgx-pg-config crate. That doesnt mean they can't be moved at a later date.
mod path_methods;
pub use path_methods::{get_target_dir, prefix_path};

#[derive(Clone)]
pub struct PgVersion {
    major: u16,
    minor: u16,
    url: Url,
}

impl PgVersion {
    pub fn new(major: u16, minor: u16, url: Url) -> PgVersion {
        PgVersion { major, minor, url }
    }
}

impl Display for PgVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

#[derive(Clone)]
pub struct PgConfig {
    version: Option<PgVersion>,
    pg_config: Option<PathBuf>,
}

impl Display for PgConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let major = self
            .major_version()
            .expect("could not determine major version");
        let minor = self
            .minor_version()
            .expect("could not determine minor version");
        let path = match self.pg_config.as_ref() {
            Some(path) => path.display().to_string(),
            None => self.version.as_ref().unwrap().url.to_string(),
        };
        write!(f, "{}.{}={}", major, minor, path)
    }
}

impl Default for PgConfig {
    fn default() -> Self {
        PgConfig {
            version: None,
            pg_config: None,
        }
    }
}

impl From<PgVersion> for PgConfig {
    fn from(version: PgVersion) -> Self {
        PgConfig {
            version: Some(version),
            pg_config: None,
        }
    }
}

impl PgConfig {
    pub fn new(pg_config: PathBuf) -> Self {
        PgConfig {
            version: None,
            pg_config: Some(pg_config),
        }
    }

    pub fn from_path() -> Self {
        PgConfig::new("pg_config".into())
    }

    pub fn is_real(&self) -> bool {
        self.pg_config.is_some()
    }

    pub fn label(&self) -> eyre::Result<String> {
        Ok(format!("pg{}", self.major_version()?))
    }

    pub fn path(&self) -> Option<PathBuf> {
        self.pg_config.clone()
    }

    pub fn parent_path(&self) -> PathBuf {
        self.path().unwrap().parent().unwrap().to_path_buf()
    }

    pub fn major_version(&self) -> eyre::Result<u16> {
        match &self.version {
            Some(version) => Ok(version.major),
            None => {
                let version_string = self.run("--version")?;
                let version_parts = version_string.split_whitespace().collect::<Vec<&str>>();
                let version = match version_parts.get(1) {
                    Some(v) => v,
                    None => {
                        return Err(eyre!("invalid version string: {}", version_string));
                    }
                };
                let version = match f64::from_str(version) {
                    Ok(f) => f,
                    Err(e) => {
                        return Err(eyre!("invalid major version number `{}`: {:?}", version, e));
                    }
                };
                Ok(version.floor() as u16)
            }
        }
    }

    pub fn minor_version(&self) -> eyre::Result<u16> {
        match &self.version {
            Some(version) => Ok(version.minor),
            None => {
                let version_string = self.run("--version")?;
                let version_parts = version_string.split_whitespace().collect::<Vec<&str>>();
                let version = match version_parts.get(1) {
                    Some(v) => v.split('.').next().unwrap(),
                    None => {
                        return Err(eyre!("invalid version string: {}", version_string));
                    }
                };
                let version = match u16::from_str(version) {
                    Ok(u) => u,
                    Err(e) => {
                        return Err(eyre!("invalid minor version number `{}`: {:?}", version, e));
                    }
                };
                Ok(version)
            }
        }
    }

    pub fn version(&self) -> eyre::Result<String> {
        let major = self.major_version()?;
        let minor = self.minor_version()?;
        let version = format!("{}.{}", major, minor);
        Ok(version)
    }

    pub fn url(&self) -> Option<&Url> {
        match &self.version {
            Some(version) => Some(&version.url),
            None => None,
        }
    }

    pub fn port(&self) -> eyre::Result<u16> {
        Ok(BASE_POSTGRES_PORT_NO + self.major_version()?)
    }

    pub fn test_port(&self) -> eyre::Result<u16> {
        Ok(BASE_POSTGRES_TESTING_PORT_NO + self.major_version()?)
    }

    pub fn host(&self) -> &'static str {
        "localhost"
    }

    pub fn bin_dir(&self) -> eyre::Result<PathBuf> {
        Ok(Path::new(&self.run("--bindir")?).to_path_buf())
    }

    pub fn postmaster_path(&self) -> eyre::Result<PathBuf> {
        let mut path = self.bin_dir()?;
        path.push("postmaster");
        Ok(path)
    }

    pub fn initdb_path(&self) -> eyre::Result<PathBuf> {
        let mut path = self.bin_dir()?;
        path.push("initdb");
        Ok(path)
    }

    pub fn createdb_path(&self) -> eyre::Result<PathBuf> {
        let mut path = self.bin_dir()?;
        path.push("createdb");
        Ok(path)
    }

    pub fn dropdb_path(&self) -> eyre::Result<PathBuf> {
        let mut path = self.bin_dir()?;
        path.push("dropdb");
        Ok(path)
    }

    pub fn psql_path(&self) -> eyre::Result<PathBuf> {
        let mut path = self.bin_dir()?;
        path.push("psql");
        Ok(path)
    }

    pub fn data_dir(&self) -> eyre::Result<PathBuf> {
        let mut path = Pgx::home()?;
        path.push(format!("data-{}", self.major_version()?));
        Ok(path)
    }

    pub fn log_file(&self) -> eyre::Result<PathBuf> {
        let mut path = Pgx::home()?;
        path.push(format!("{}.log", self.major_version()?));
        Ok(path)
    }

    pub fn includedir_server(&self) -> eyre::Result<PathBuf> {
        Ok(self.run("--includedir-server")?.into())
    }

    pub fn pkglibdir(&self) -> eyre::Result<PathBuf> {
        Ok(self.run("--pkglibdir")?.into())
    }

    pub fn sharedir(&self) -> eyre::Result<PathBuf> {
        Ok(self.run("--sharedir")?.into())
    }

    pub fn extension_dir(&self) -> eyre::Result<PathBuf> {
        let mut path = self.sharedir()?;
        path.push("extension");
        Ok(path)
    }

    fn run(&self, arg: &str) -> eyre::Result<String> {
        let pg_config = self.pg_config.clone().unwrap_or_else(|| {
            std::env::var("PG_CONFIG")
                .unwrap_or_else(|_| "pg_config".to_string())
                .into()
        });

        match Command::new(&pg_config).arg(arg).output() {
            Ok(output) => Ok(String::from_utf8(output.stdout).unwrap().trim().to_string()),
            Err(e) => match e.kind() {
                ErrorKind::NotFound => {
                    Err(e).wrap_err_with(|| format!("Unable to find `{}`", "pg_config".yellow()))
                }
                _ => Err(e.into()),
            },
        }
    }
}

pub struct Pgx {
    pg_configs: Vec<PgConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConfigToml {
    configs: HashMap<String, PathBuf>,
}

pub enum PgConfigSelector<'a> {
    All,
    Specific(&'a str),
}

impl<'a> PgConfigSelector<'a> {
    pub fn new(label: &'a str) -> Self {
        if label == "all" {
            PgConfigSelector::All
        } else {
            PgConfigSelector::Specific(label)
        }
    }
}

impl Pgx {
    pub fn new() -> Self {
        Pgx { pg_configs: vec![] }
    }

    pub fn from_config() -> eyre::Result<Self> {
        match std::env::var("PGX_PG_CONFIG_PATH") {
            Ok(pg_config) => {
                // we have an environment variable that tells us the pg_config to use
                let mut pgx = Pgx::new();
                pgx.push(PgConfig::new(pg_config.into()));
                Ok(pgx)
            }
            Err(_) => {
                // we'll get what we need from cargo-pgx' config.toml file
                let path = Pgx::config_toml()?;
                if !path.exists() {
                    return Err(eyre!(
                        "{} not found.  Have you run `{}` yet?",
                        path.display(),
                        "cargo pgx init".bold().yellow()
                    ));
                }

                match toml::from_str::<ConfigToml>(&std::fs::read_to_string(&path)?) {
                    Ok(configs) => {
                        let mut pgx = Pgx::new();

                        for (_, v) in configs.configs {
                            pgx.push(PgConfig::new(v));
                        }
                        Ok(pgx)
                    }
                    Err(e) => {
                        Err(e).wrap_err_with(|| format!("Could not read `{}`", path.display()))
                    }
                }
            }
        }
    }

    pub fn push(&mut self, pg_config: PgConfig) {
        self.pg_configs.push(pg_config);
    }

    pub fn iter(
        &self,
        which: PgConfigSelector,
    ) -> impl std::iter::Iterator<Item = eyre::Result<&PgConfig>> {
        match which {
            PgConfigSelector::All => {
                let mut configs = self.pg_configs.iter().collect::<Vec<_>>();
                configs.sort_by(|a, b| {
                    a.major_version()
                        .expect("no major version")
                        .cmp(&b.major_version().expect("no major version"))
                });

                configs
                    .into_iter()
                    .map(|c| Ok(c))
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            PgConfigSelector::Specific(label) => vec![self.get(label)].into_iter(),
        }
    }

    #[tracing::instrument(level = "error", skip(self))]
    pub fn get(&self, label: &str) -> eyre::Result<&PgConfig> {
        for pg_config in self.pg_configs.iter() {
            if pg_config.label()? == label {
                return Ok(pg_config);
            }
        }
        Err(eyre!("Postgres `{}` is not managed by pgx", label))
    }

    pub fn home() -> Result<PathBuf, std::io::Error> {
        std::env::var("PGX_HOME").map_or_else(
            |_| {
                let mut dir = match dirs::home_dir() {
                    Some(dir) => dir,
                    None => {
                        return Err(std::io::Error::new(
                            ErrorKind::NotFound,
                            "You don't seem to have a home directory",
                        ));
                    }
                };
                dir.push(".pgx");
                if !dir.exists() {
                    if let Err(e) = std::fs::create_dir_all(&dir) {
                        return Err(std::io::Error::new(
                            ErrorKind::InvalidInput,
                            format!("could not create PGX_HOME at `{}`: {:?}", dir.display(), e),
                        ));
                    }
                }

                Ok(dir)
            },
            |v| Ok(v.into()),
        )
    }

    /// Get the postmaster stub directory
    ///
    /// We isolate postmaster stubs to an independent directory instead of alongside the postmaster
    /// because in the case of `cargo pgx install` the `pg_config` may not necessarily be one managed
    /// by pgx.
    pub fn postmaster_stub_dir() -> Result<PathBuf, std::io::Error> {
        let mut stub_dir = Self::home()?;
        stub_dir.push("postmaster_stubs");
        Ok(stub_dir)
    }

    pub fn config_toml() -> Result<PathBuf, std::io::Error> {
        let mut path = Pgx::home()?;
        path.push("config.toml");
        Ok(path)
    }
}

pub const SUPPORTED_MAJOR_VERSIONS: &[u16] = &[10, 11, 12, 13, 14];

pub fn createdb(
    pg_config: &PgConfig,
    dbname: &str,
    is_test: bool,
    if_not_exists: bool,
) -> eyre::Result<bool> {
    if if_not_exists && does_db_exist(pg_config, dbname)? {
        return Ok(false);
    }

    println!("{} database {}", "     Creating".bold().green(), dbname);
    let mut command = Command::new(pg_config.createdb_path()?);
    command
        .env_remove("PGDATABASE")
        .env_remove("PGHOST")
        .env_remove("PGPORT")
        .env_remove("PGUSER")
        .arg("-h")
        .arg(pg_config.host())
        .arg("-p")
        .arg(if is_test {
            pg_config.test_port()?.to_string()
        } else {
            pg_config.port()?.to_string()
        })
        .arg(dbname)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let command_str = format!("{:?}", command);

    let child = command.spawn().wrap_err_with(|| {
        format!(
            "Failed to spawn process for creating database using command: '{}': ",
            command_str
        )
    })?;

    let output = child.wait_with_output().wrap_err_with(|| {
        format!(
            "Failed waiting for spawned process attempting to create database using command: '{}': ",
            command_str
        )
    })?;

    if !output.status.success() {
        return Err(eyre!(
            "problem running createdb: {}\n\n{}{}",
            command_str,
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap()
        ));
    }

    Ok(true)
}

fn does_db_exist(pg_config: &PgConfig, dbname: &str) -> eyre::Result<bool> {
    let mut command = Command::new(pg_config.psql_path()?);
    command
        .arg("-XqAt")
        .env_remove("PGUSER")
        .arg("-h")
        .arg(pg_config.host())
        .arg("-p")
        .arg(pg_config.port()?.to_string())
        .arg("template1")
        .arg("-c")
        .arg(&format!(
            "select count(*) from pg_database where datname = '{}';",
            dbname.replace("'", "''")
        ))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let command_str = format!("{:?}", command);
    let output = command.output()?;

    if !output.status.success() {
        return Err(eyre!(
            "problem checking if database '{}' exists: {}\n\n{}{}",
            dbname,
            command_str,
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap()
        ));
    } else {
        let count = i32::from_str(&String::from_utf8(output.stdout).unwrap().trim())
            .wrap_err("result is not a number")?;
        Ok(count > 0)
    }
}
