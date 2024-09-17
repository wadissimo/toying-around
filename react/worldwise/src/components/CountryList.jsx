import CountryItem from "./CountryItem";
import styles from "./CountryList.module.css";
import Message from "./Message";
import Spinner from "./Spinner";

export default function CountryList({ cities, isLoading }) {
  if (isLoading) {
    return <Spinner />;
  }

  if (!cities.length) {
    return (
      <Message message="Add your first city by clicking a city on the map." />
    );
  }

  const { countries } = cities.reduce(
    (arr, city) => {
      if (arr.uniqueCountries.includes(city.country)) {
        return arr;
      } else {
        arr.uniqueCountries = [...arr.uniqueCountries];
        return {
          uniqueCountries: [...arr.uniqueCountries, city.country],
          countries: [
            ...arr.countries,
            { country: city.country, emoji: city.emoji },
          ],
        };
      }
    },
    { uniqueCountries: [], countries: [] }
  );

  return (
    <ul className={styles.countryList}>
      {countries.map((country) => (
        <CountryItem country={country} key={country.country} />
      ))}
    </ul>
  );
}
